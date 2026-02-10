"use client";

/*
 High-level summary of recent updates:
 - Moved conversation history into a docked right-side panel (open/close, transitions)
 - Added controls to create, select, and delete conversations from the history panel
 - Replaced top chat-tab UI with the right-side history panel and added a floating handle
 - Fixed message send/streaming logic and integrated localStorage persistence for conversations
 - Wire-up to open the history panel via the Sidebar 'Chats' button

 These comments are intentionally brief; refer to `CHANGES.md` for a repo-level summary.
*/

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { fileApi, chatApi } from '@/lib/api';
import { FiSend, FiPlus, FiMenu, FiX, FiChevronDown, FiDatabase, FiDownload } from 'react-icons/fi';
import toast, { Toaster } from 'react-hot-toast';
import Sidebar from '@/components/Sidebar';
import ChatHistoryPanel from '@/components/ChatHistoryPanel';
// Lightweight, safe Markdown renderer for bold, italics, lists, and line breaks.
const escapeHtml = (unsafe: string) =>
  unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');

const renderSimpleMarkdown = (text: string) => {
  if (!text) return '';
  // Escape HTML first
  let out = escapeHtml(text);

  // Bold **text**
  out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic *text* (avoid interfering with bold)
  out = out.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
  // Simple unordered lists: lines starting with - or *
  const lines = out.split(/\r?\n/);
  let inList = false;
  const processed: string[] = [];
  for (let line of lines) {
    if (/^\s*[-*]\s+/.test(line)) {
      if (!inList) {
        processed.push('<ul>');
        inList = true;
      }
      processed.push('<li>' + line.replace(/^\s*[-*]\s+/, '') + '</li>');
    } else {
      if (inList) {
        processed.push('</ul>');
        inList = false;
      }
      // Preserve paragraphs / line breaks
      if (line.trim() === '') processed.push('<br/>');
      else processed.push('<p>' + line + '</p>');
    }
  }
  if (inList) processed.push('</ul>');
  return processed.join('\n');
};

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  files?: string[];
  timestamp: Date;
  duration?: number; // Response duration in seconds
  chunks?: any[]; // Retrieved document chunks
  retrievalTime?: number; // Retrieval time in seconds
  breakdown?: any; // Timing breakdown
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}

// Simple directed graph visualization for Star Topology (Doc -> Entities)
const SimpleGraphView = ({ data }: { data: any }) => {
  if (!data || !data.nodes || !data.relationships) return null;

  const width = 500;
  const height = 300;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = 100;

  // Identify root node (Document) vs Entity nodes
  let centerNode = data.nodes.find((n: any) => n.labels.includes('Document')) || data.nodes[0];
  const entityNodes = data.nodes.filter((n: any) => n.id !== centerNode.id);

  // Calculate positions
  const nodesWithPos = [
    { ...centerNode, x: centerX, y: centerY, type: 'root' },
    ...entityNodes.map((node: any, i: number) => {
      const angle = (i / entityNodes.length) * 2 * Math.PI;
      return {
        ...node,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        type: 'entity'
      };
    })
  ];

  return (
    <div className="border rounded bg-white overflow-hidden my-2 border-gray-100 shadow-sm relative">
      <div className="absolute top-2 left-2 text-xs font-bold text-gray-400 uppercase tracking-widest z-10">Knowledge Graph</div>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="w-full">
        {/* Edges */}
        {nodesWithPos.filter((n: any) => n.type !== 'root').map((node: any) => (
          <line
            key={`edge-${node.id}`}
            x1={centerX}
            y1={centerY}
            x2={node.x}
            y2={node.y}
            stroke="#e5e7eb"
            strokeWidth="2"
          />
        ))}

        {/* Nodes */}
        {nodesWithPos.map((node: any) => (
          <g key={node.id}>
            <circle
              cx={node.x}
              cy={node.y}
              r={node.type === 'root' ? 25 : 18}
              fill={node.type === 'root' ? '#d1fae5' : '#eff6ff'}
              stroke={node.type === 'root' ? '#059669' : '#3b82f6'}
              strokeWidth="2"
            />
            <text
              x={node.x}
              y={node.y + (node.type === 'root' ? 40 : 30)}
              fontSize="10"
              textAnchor="middle"
              fill="#374151"
              className="font-medium"
            >
              {node.id.split(':').pop()?.substring(0, 15)}
            </text>
            {/* Icon/Label inside circle */}
            <text
              x={node.x}
              y={node.y + 4}
              fontSize={node.type === 'root' ? "12" : "10"}
              textAnchor="middle"
              fill={node.type === 'root' ? '#065f46' : '#1e40af'}
              fontWeight="bold"
            >
              {node.type === 'root' ? 'DOC' : 'ENT'}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default function ChatPage() {
  const router = useRouter();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const didAutoCreateRef = useRef(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  const [showFileDropdown, setShowFileDropdown] = useState(false);
  const [hasModels, setHasModels] = useState<boolean | null>(null); // null = loading, true/false = checked
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Use a fixed user_id for single-user mode
  const USER_ID = 'default_user';

  useEffect(() => {
    const init = async () => {
      await checkModels(); // Wait for model check first
      fetchFiles();
      loadConversations();
      setIsInitialized(true);
      // If navigation requested opening chat panel, clear flag and open panel
      try {
        const val = localStorage.getItem('chatPanelOpen');
        if (val === 'true') {
          setShowHistoryPanel(true);
          localStorage.removeItem('chatPanelOpen');
        }
      } catch (err) {
        // ignore
      }
    };

    // Reset initialization flag when component mounts/remounts
    setIsInitialized(false);
    init();
  }, []); // Empty array ensures this runs on mount/remount

  // Also reload conversations when returning to page via client-side navigation
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        // Page became visible again, reload conversations to sync with localStorage
        loadConversations();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  // Create initial conversation only after loading is complete and no conversations exist
  useEffect(() => {
    // Auto-create an initial conversation only once after initialization.
    // This avoids immediately recreating a "New Chat" after the user deletes
    // the last conversation (user expects it to be removed).
    if (isInitialized && conversations.length === 0 && !currentConversation && !didAutoCreateRef.current) {
      createNewConversation();
      didAutoCreateRef.current = true;
    }
  }, [isInitialized, conversations.length, currentConversation]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentConversation?.messages]);

  const fetchFiles = async () => {
    try {
      const response = await fileApi.getUploads(USER_ID);
      setFiles(
        response.files
          .filter((f: any) => f.available === 1)
          .map((f: any) => f.filename)
      );
    } catch (error) {
      toast.error('Failed to load files');
    }
  };

  const checkModels = async () => {
    try {
      const response = await fetch('http://localhost:8002/models');
      const data = await response.json();
      // Check if we have any models installed or active models configured
      const hasLLM = (data.models && data.models.some((m: any) => m.type === 'llm')) ||
        (data.active && data.active.llm);
      setHasModels(hasLLM);
    } catch (error) {
      console.error('Failed to check models:', error);
      setHasModels(false);
    }
  };

  const loadConversations = () => {
    // Load from localStorage for now
    const saved = localStorage.getItem(`conversations_${USER_ID}`);
    if (saved) {
      const parsed = JSON.parse(saved);
      setConversations(parsed);
      if (parsed.length > 0) {
        setCurrentConversation(parsed[0]);
      }
    }
  };

  const createNewConversation = () => {
    const newConv: Conversation = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
    };
    setConversations([newConv, ...conversations]);
    setCurrentConversation(newConv);
    setSelectedFiles(new Set());
  };

  const saveConversations = (convs: Conversation[]) => {
    localStorage.setItem(`conversations_${USER_ID}`, JSON.stringify(convs));
  };
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;
    setIsLoading(true);

    try {
      // Ensure we have a conversation and track it properly
      let finalConv = currentConversation;
      let currentConvs = conversations;

      if (!finalConv) {
        finalConv = {
          id: Date.now().toString(),
          title: 'New Chat',
          messages: [],
          createdAt: new Date(),
        };
        currentConvs = [finalConv, ...conversations];
        setConversations(currentConvs);
      }

      // Append user message
      const userMessage: Message = {
        id: `${Date.now()}_u`,
        role: 'user',
        content: query,
        files: Array.from(selectedFiles),
        timestamp: new Date(),
      };

      finalConv = { ...finalConv, messages: [...finalConv.messages, userMessage] };
      setCurrentConversation(finalConv);

      // Update conversations array and save immediately after user message
      currentConvs = currentConvs.map((c) => (c.id === finalConv!.id ? finalConv! : c));
      setConversations(currentConvs);
      saveConversations(currentConvs);

      setQuery('');

      // Prepare assistant placeholder
      let assistantContent = '';
      const assistantMessage: Message = {
        id: `${Date.now()}_a`,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
      };

      finalConv = { ...finalConv, messages: [...finalConv.messages, assistantMessage] };
      setCurrentConversation(finalConv);

      // Update conversations array with placeholder
      currentConvs = currentConvs.map((c) => (c.id === finalConv!.id ? finalConv! : c));
      setConversations(currentConvs);

      // Stream the response from the backend
      const result = await chatApi.ragChat(
        query,
        USER_ID,
        Array.from(selectedFiles),
        (chunk: string) => {
          assistantContent += chunk;
          assistantMessage.content = assistantContent;
          const streamConv = {
            ...finalConv,
            messages: [...finalConv.messages.slice(0, -1), assistantMessage],
          };
          setCurrentConversation(streamConv);

          // Update conversations array during streaming using functional update
          setConversations(prevConvs => {
            const updated = prevConvs.map((c) => (c.id === finalConv!.id ? streamConv : c));
            // Save during streaming to ensure persistence
            saveConversations(updated);
            return updated;
          });
        }
      );

      if (result.duration) assistantMessage.duration = result.duration;
      if (result.chunks) assistantMessage.chunks = result.chunks;
      if (result.retrievalTime) assistantMessage.retrievalTime = result.retrievalTime;
      if (result.breakdown) assistantMessage.breakdown = result.breakdown;

      finalConv = { ...finalConv, messages: [...finalConv.messages.slice(0, -1), assistantMessage] };
      setCurrentConversation(finalConv);

      const updatedConvsFinal = conversations.map((c) => (c.id === finalConv!.id ? finalConv! : c));
      setConversations(updatedConvsFinal);
      saveConversations(updatedConvsFinal);
    } catch (error) {
      toast.error('Failed to send message');
      // Remove the user message on error
      if (currentConversation) {
        const updatedConv = {
          ...currentConversation,
          messages: currentConversation.messages.slice(0, -1),
        };
        setCurrentConversation(updatedConv);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    // No authentication in single-user mode
    router.push('/');
  };

  const toggleFileSelection = (filename: string) => {
    const newSelected = new Set(selectedFiles);
    if (newSelected.has(filename)) {
      newSelected.delete(filename);
    } else {
      newSelected.add(filename);
    }
    setSelectedFiles(newSelected);
  };

  const handleDeleteConversation = (id: string) => {
    // Normalize IDs to string to avoid mismatches between numeric and string ids
    const remaining = conversations.filter((c) => String(c.id) !== String(id));
    setConversations(remaining);
    saveConversations(remaining);

    // Notify user
    toast.success('Conversation deleted');

    if (currentConversation && String(currentConversation.id) === String(id)) {
      // Select next available conversation or clear
      if (remaining.length > 0) {
        setCurrentConversation(remaining[0]);
      } else {
        setCurrentConversation(null);
      }
    }
  };

  return (
    <div className="flex w-full h-screen bg-white">
      <Sidebar />

      <main className="flex-1 flex flex-col h-screen relative">
        <Toaster />

        {/* Top Header */}
        <header className="h-16 border-b border-gray-200 flex items-center justify-between px-6 bg-white z-10">
          <div className="flex items-center gap-2 text-sm text-gray-500 cursor-pointer hover:bg-gray-50 p-2 rounded-lg">
            <div className="w-5 h-5 rounded bg-green-600" />
            <span className="font-medium text-gray-700">Gemma-3</span>
            <FiChevronDown />
          </div>

          <button
            onClick={() => router.push('/dashboard')}
            className="flex items-center gap-2 px-3 py-1.5 bg-white border border-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition shadow-sm"
          >
            <FiDatabase className="text-green-600" />
            Local Documents
          </button>
        </header>

        {/* Messaging Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 scroll-smooth">
          {hasModels === null ? (
            /* Loading state - checking for models */
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-green-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-gray-500">Loading...</p>
              </div>
            </div>
          ) : !hasModels ? (
            /* No Models Installed - Welcome Screen */
            <div className="h-full flex flex-col items-center justify-center px-8 pb-20">
              <div className="max-w-lg text-center">
                <div className="w-20 h-20 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-6">
                  <FiDownload className="w-10 h-10 text-gray-400" />
                </div>

                <h2 className="text-3xl font-bold text-gray-800 mb-3">No Model Installed</h2>
                <p className="text-gray-500 mb-8 leading-relaxed">
                  Your Personal Assistant requires that you install at least one model to get started.
                  Choose from a variety of AI models optimized for different tasks.
                </p>

                <Link
                  href="/models"
                  className="inline-flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition shadow-md"
                >
                  <FiDownload />
                  Install a Model
                </Link>

                <p className="text-xs text-gray-400 mt-6">
                  Models are downloaded from Ollama and run locally on your machine
                </p>
              </div>
            </div>
          ) : !currentConversation || currentConversation.messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-300 pb-20">
              <h1 className="text-4xl font-bold text-gray-800 mb-8 tracking-tight">SiloQ</h1>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {currentConversation.messages.map((msg) => (
                <div key={msg.id} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0 mt-1">
                      AI
                    </div>
                  )}
                  <div className={`max-w-[85%] rounded-2xl px-5 py-3 ${msg.role === 'user'
                    ? 'bg-gray-100 text-gray-900 rounded-tr-none'
                    : 'bg-white border border-gray-200 text-gray-900 shadow-sm'
                    }`}>
                    {msg.role === 'assistant' ? (
                      <div className="prose prose-sm max-w-full" dangerouslySetInnerHTML={{ __html: renderSimpleMarkdown(msg.content || '...') }} />
                    ) : (
                      <p className="whitespace-pre-wrap leading-relaxed break-words">{msg.content || '...'}</p>
                    )}
                    {msg.files && msg.files.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-200/50 flex gap-2 flex-wrap">
                        {msg.files.map(f => (
                          <span key={f} className="text-xs text-gray-500 bg-gray-50 px-2 py-1 rounded border border-gray-200">
                            📄 {f}
                          </span>
                        ))}
                      </div>
                    )}
                    {msg.role === 'assistant' && msg.chunks && msg.chunks.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200/50">
                        <details className="text-xs">
                          <summary className="cursor-pointer text-gray-500 hover:text-gray-700 font-medium mb-2">
                            📚 Retrieved {msg.chunks.length} document chunks ({msg.retrievalTime?.toFixed(2)}s)
                          </summary>
                          <div className="space-y-2 mt-2">
                            {msg.chunks.map((chunk: any, idx: number) => (
                              <div key={idx} className="bg-gray-50 rounded p-3 border border-gray-200">
                                <div className="flex items-start gap-2 mb-2">
                                  <span className="font-mono text-green-600 font-bold text-sm">#{chunk.rank}</span>
                                  <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-1">
                                      <span className="text-blue-600 font-medium">📄 {chunk.source || chunk.filename}</span>
                                    </div>
                                    {chunk.filename !== chunk.source && chunk.filename !== 'unknown' && (
                                      <span className="text-gray-500 text-xs">Chunk: {chunk.filename}</span>
                                    )}

                                    {/* Score Badge */}
                                    {chunk.metadata?.score && (
                                      <span className="ml-2 text-[10px] bg-green-100 text-green-800 px-1.5 py-0.5 rounded-full border border-green-200">
                                        Score: {chunk.metadata.score.toFixed(3)}
                                      </span>
                                    )}
                                  </div>
                                </div>

                                {/* Render Graph if available, else text */}
                                {chunk.metadata?.graph_data ? (
                                  <div className="mb-2">
                                    <SimpleGraphView data={chunk.metadata.graph_data} />
                                    <details className="mt-2">
                                      <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">Show text content</summary>
                                      <p className="text-gray-700 leading-relaxed text-sm mt-2 pl-2 border-l-2 border-green-200">
                                        {chunk.content}
                                      </p>
                                    </details>
                                  </div>
                                ) : (
                                  <p className="text-gray-700 leading-relaxed">
                                    {chunk.content}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        </details>
                      </div>
                    )}
                    {msg.role === 'assistant' && msg.duration && (
                      <div className="mt-2 pt-2 border-t border-gray-200/50">
                        <span className="text-xs text-gray-400">
                          ⏱️ Response time: {msg.duration.toFixed(1)}s
                          {msg.breakdown && (
                            <span className="ml-2">
                              (retrieval: {msg.breakdown.retrieval}s, LLM: {msg.breakdown.llm}s)
                            </span>
                          )}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
              {isLoading && (
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0 animate-pulse">
                    AI
                  </div>
                  <div className="text-gray-500 italic text-sm py-2">
                    Thinking...
                  </div>
                </div>
              )}
            </div>
          )
          }
        </div >

        {/* Input Area */}
        < div className="p-4 bg-white" >
          <div className="max-w-3xl mx-auto relative group">
            {selectedFiles.size > 0 && (
              <div className="absolute -top-12 left-0 flex gap-2 overflow-x-auto w-full pb-2 px-1">
                {Array.from(selectedFiles).map(f => (
                  <div key={f} className="flex items-center gap-1 text-xs bg-green-50 text-green-700 px-3 py-1.5 rounded-full border border-green-100 whitespace-nowrap shadow-sm">
                    <span className="truncate max-w-[150px]">{f}</span>
                    <button onClick={() => toggleFileSelection(f)} className="hover:text-green-900 ml-1">×</button>
                  </div>
                ))}
              </div>
            )}

            <form onSubmit={handleSendMessage} className="relative shadow-sm rounded-2xl border border-gray-300 focus-within:ring-1 focus-within:ring-green-500 focus-within:border-green-500 transition-all bg-white">
              <input
                className="w-full py-4 pl-4 pr-32 bg-transparent outline-none text-gray-700 placeholder-gray-400 rounded-2xl"
                placeholder="Send a message..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={isLoading}
              />

              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setShowFileDropdown(!showFileDropdown)}
                    className={`p-2 rounded-lg transition ${selectedFiles.size > 0 ? 'text-green-600 bg-green-50' : 'text-gray-400 hover:text-gray-600'}`}
                    title="Attach Local Documents"
                  >
                    <FiDatabase size={18} />
                  </button>

                  {showFileDropdown && (
                    <div className="absolute bottom-full right-0 mb-4 w-72 bg-white rounded-xl shadow-2xl border border-gray-100 p-3 max-h-80 overflow-y-auto z-20">
                      <div className="flex justify-between items-center mb-2 px-1">
                        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Your Documents</h3>
                        <button onClick={() => router.push('/dashboard')} className="text-xs text-green-600 hover:underline">+ Add Folder</button>
                      </div>
                      {files.length === 0 ? (
                        <div className="text-center py-4">
                          <p className="text-xs text-gray-400">No documents found.</p>
                        </div>
                      ) : (
                        <div className="space-y-1">
                          {files.map(file => (
                            <div
                              key={file}
                              className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer text-sm transition ${selectedFiles.has(file) ? 'bg-green-50 text-green-800' : 'hover:bg-gray-50 text-gray-700'}`}
                              onClick={() => toggleFileSelection(file)}
                            >
                              <div className={`w-4 h-4 rounded border flex items-center justify-center transition ${selectedFiles.has(file) ? 'bg-green-500 border-green-500' : 'border-gray-300'}`}>
                                {selectedFiles.has(file) && <div className="w-2 h-2 bg-white rounded-sm" />}
                              </div>
                              <span className="truncate flex-1">{file}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <button
                  type="submit"
                  disabled={!query.trim() || isLoading}
                  className={`p-2 rounded-lg transition shadow-sm ${!query.trim() || isLoading ? 'bg-gray-100 text-gray-400' : 'bg-green-600 text-white hover:bg-green-700'}`}
                >
                  <FiSend size={18} />
                </button>
              </div>
            </form>
            <div className="text-center mt-3 text-xs text-gray-400 font-light">
              SiloQ can make mistakes. Please review critical information.
            </div>
          </div>
        </div >
      </main >

      {/* Docked history panel as a flex sibling with width transition */}
      < div className={`h-full transition-all duration-300 ease-in-out ${showHistoryPanel ? 'w-96' : 'w-0'} overflow-hidden`}>
        <div className="h-full flex flex-col">
          {showHistoryPanel && (
            <ChatHistoryPanel
              conversations={conversations}
              onSelect={(c) => { setCurrentConversation(c); setShowHistoryPanel(false); }}
              onClose={() => setShowHistoryPanel(false)}
              onNew={() => { createNewConversation(); setShowHistoryPanel(false); }}
              onDelete={(id) => handleDeleteConversation(id)}
            />
          )}
        </div>
      </div >

      {/* Floating handle to re-open the history panel after user closes it */}
      {
        !showHistoryPanel && (
          <button
            aria-label="Open conversation history"
            title="Open conversation history"
            onClick={() => setShowHistoryPanel(true)}
            className="fixed right-0 top-1/2 -translate-y-1/2 mr-2 z-40 bg-white border border-gray-200 rounded-l-full px-3 py-2 shadow hover:bg-gray-50 focus:outline-none"
          >
            Chats
          </button>
        )
      }
    </div >
  );
}

