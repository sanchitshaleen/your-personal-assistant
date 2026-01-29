'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { fileApi } from '@/lib/api';
import Sidebar from '@/components/Sidebar';
import { FiUpload, FiTrash2, FiSearch, FiFileText, FiCalendar, FiDatabase, FiRefreshCw, FiFolder } from 'react-icons/fi';
import toast, { Toaster } from 'react-hot-toast';

interface UploadedFile {
    filename: string;
    created_at: string;
    available: number;
    source?: 'manual' | 'google_drive' | 'local';
    size?: string;
    embedding_status?: string;
    embedding_model?: string;
}

export default function DashboardPage() {
    const router = useRouter();
    const [files, setFiles] = useState<UploadedFile[]>([]);
    const [taskMap, setTaskMap] = useState<Record<string, string>>({}); // Map filename -> task_id
    const [progressMap, setProgressMap] = useState<Record<string, number>>({}); // Map filename -> percentage
    
    // Use a fixed user_id for single-user mode
    const USER_ID = 'default_user';

    const [isHydrated, setIsHydrated] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [uploadProgress, setUploadProgress] = useState<{ 
        step: number; 
        total: number; 
        message: string;
        percentage?: number;
        taskId?: string;
    } | null>(null);

    // Collapsed folders state
    const [expandedFolders, setExpandedFolders] = useState<Record<string, boolean>>({});
    
    // Embedding model selection
    const [selectedModel, setSelectedModel] = useState<string>('nomic-embed-text');
    const [availableModels, setAvailableModels] = useState<{ id: string; name: string }[]>([
        { id: 'nomic-embed-text', name: 'Nomic Embed Text v1.5 (8k context)' },
        { id: 'mxbai-embed-large:latest', name: 'Mxbai Embed Large (Best Precision)' },
    ]);

    // Group files by folder
    const groupedFiles = files.reduce((acc, file) => {
        const parts = file.filename.split('/');
        const isFolder = parts.length > 1;
        
        if (isFolder) {
            const folderName = parts[0];
            if (!acc[folderName]) acc[folderName] = [];
            acc[folderName].push(file);
        } else {
            if (!acc['root']) acc['root'] = [];
            acc['root'].push(file);
        }
        return acc;
    }, {} as Record<string, UploadedFile[]>);

    const toggleFolder = (folderName: string) => {
        setExpandedFolders(prev => ({
            ...prev,
            [folderName]: !prev[folderName]
        }));
    };

    useEffect(() => {
        setIsHydrated(true);
    }, []);

    useEffect(() => {
        if (!isHydrated) return;
        fetchFiles();
        fetchModels();
        
        const interval = setInterval(() => {
            fetchFiles();
            // Poll for task progress
            Object.entries(taskMap).forEach(([filename, taskId]) => {
                // Only poll if file is not completed according to file list
                const file = files.find(f => f.filename === filename);
                if (file && file.embedding_status !== 'completed' && file.embedding_status !== 'failed') {
                    checkProgress(filename, taskId);
                }
            });
        }, 3000);
        return () => clearInterval(interval);
    }, [isHydrated, router, taskMap, files]);

    const checkProgress = async (filename: string, taskId: string) => {
        try {
            const status = await fileApi.getTaskStatus(taskId);
            if (status.progress !== undefined) {
                setProgressMap(prev => ({ ...prev, [filename]: status.progress }));
            }
            // If completed or failed, remove from taskMap to stop polling
            if (status.status === 'completed' || status.status === 'failed') {
                setTaskMap(prev => {
                    const newMap = { ...prev };
                    delete newMap[filename];
                    return newMap;
                });
            }
        } catch (e) {
            console.error("Error checking progress", e);
        }
    };

    const fetchFiles = async () => {
        try {
            const response = await fileApi.getUploads(USER_ID);
            setFiles(response.files || []);
        } catch (error) {
            console.error('Failed to load files', error);
        }
    };
    
    const fetchModels = async () => {
        try {
            const res = await fileApi.getModels();
            if (res.models) {
                // Filter for embedding models
                const embeddingModels = res.models.filter((m: any) => m.type === 'embedding');
                
                if (embeddingModels.length > 0) {
                    setAvailableModels(embeddingModels.map((m: any) => ({
                        id: m.name,
                        name: m.name
                    })));
                    // Set default if current selection is not in list (optional)
                    // if (!embeddingModels.find((m: any) => m.name === selectedModel)) {
                    //     setSelectedModel(embeddingModels[0].name);
                    // }
                }
            }
        } catch (e) {
            console.error("Failed to fetch models", e);
        }
    };

    // Helper to process a single file (upload + trigger embed)
    const processFile = async (file: File) => {
        try {
            // Prefer webkitRelativePath if available (for folder structure), else fallback to name
            const relativePath = (file as any).webkitRelativePath;
            const fileNameToSend = relativePath || file.name;

            const uploadRes = await fileApi.uploadFile(USER_ID, file, fileNameToSend);
            // Filename might be modified (e.g., flattened or prefixed)
            const serverFilename = uploadRes.message;
            
            const embedRes = await fileApi.embedFile(USER_ID, serverFilename, selectedModel);
            if (embedRes.task_id) {
                setTaskMap(prev => ({ ...prev, [serverFilename]: embedRes.task_id }));
                setProgressMap(prev => ({ ...prev, [serverFilename]: 0 }));
            }

            return { status: 'success', filename: file.name };
        } catch (error: any) {
            console.error(`Error processing ${file.name}:`, error);
            return { status: 'error', filename: file.name, error: error.message };
        }
    };

    const handleFileUpload = async (fileList: FileList) => {
         if (!fileList) return;

         const files = Array.from(fileList).filter(file => {
            const ext = file.name.split('.').pop()?.toLowerCase();
            return ['pdf', 'txt', 'md'].includes(ext || '');
         });

         if (files.length === 0) {
             toast.error("No valid files selected. Supported: PDF, TXT, MD");
             return;
         }

         setUploadProgress({ step: 1, total: files.length, message: 'Starting upload...', percentage: 0 });

         let processed = 0;
         for (const file of files) {
             setUploadProgress({ 
                 step: processed + 1, 
                 total: files.length, 
                 message: `Uploading ${file.name}...`, 
                 percentage: Math.round((processed / files.length) * 100) 
             });
             
             await processFile(file);
             processed++;
         }

         setUploadProgress(null);
         toast.success(`Uploaded ${files.length} file(s). Processing in background.`);
         fetchFiles();
    };

    const handleFolderUpload = async (fileList: FileList) => {
         if (!fileList) return;

         const files = Array.from(fileList).filter(file => {
            const ext = file.name.split('.').pop()?.toLowerCase();
            return ['pdf', 'txt', 'md'].includes(ext || '');
         });

         if (files.length === 0) {
             toast.error("No valid files found in folder");
             return;
         }
         
         // Extract the source folder path from the first file
         let sourceFolderPath: string | null = null;
         if (files.length > 0) {
             const firstFile = files[0] as any;
             if (firstFile.webkitRelativePath) {
                 // Try to get the actual folder path from the file input
                 // Note: Browser security prevents direct access to full paths
                 // We'll prompt the user to enter it
                 const folderName = firstFile.webkitRelativePath.split('/')[0];
                 sourceFolderPath = prompt(
                     `To enable automatic file monitoring, please enter the full path to the "${folderName}" folder:\n\nExample: /Users/yourusername/Documents/${folderName}`,
                     `/Users/neetikasaxena/Documents/${folderName}`
                 );
             }
         }
         
         setUploadProgress({ step: 1, total: files.length, message: 'Starting upload...', percentage: 0 });

         let processed = 0;
         for (const file of files) {
             setUploadProgress({ 
                 step: processed + 1, 
                 total: files.length, 
                 message: `Uploading ${file.name}...`, 
                 percentage: Math.round((processed / files.length) * 100) 
             });
             
             await processFile(file);
             processed++;
         }

         setUploadProgress(null);
         
         toast.success(`Uploaded ${files.length} files. Processing in background.`);
         
         fetchFiles();
    };

    const handleDelete = async (filename: string) => {
        if (!confirm(`Are you sure you want to delete ${filename}?`)) return;
        try {
            await fileApi.deleteFile(USER_ID, filename);
            toast.success('File deleted');
            fetchFiles();
        } catch (error: any) {
            toast.error(error.message || 'Delete failed');
        }
    };

    const handleRebuild = async (filename: string) => {
        try {
            const res = await fileApi.embedFile(USER_ID, filename, selectedModel);
            if (res.task_id) {
                setTaskMap(prev => ({ ...prev, [filename]: res.task_id }));
                setProgressMap(prev => ({ ...prev, [filename]: 0 }));
            }
            toast.success('Rebuild triggered');
            fetchFiles();
        } catch (error: any) {
             toast.error(error.message || 'Rebuild failed');
        }
    };

    const handleDeleteAll = async () => {
        if (!confirm(`⚠️ DELETE ALL FILES?\n\nThis will permanently delete all ${files.length} files and their embeddings.\n\nThis action cannot be undone!`)) return;
        
        try {
            toast.loading('Deleting all files...');
            const res = await fileApi.deleteAllFiles(USER_ID);
            toast.dismiss();
            toast.success(`Deleted ${res.files_deleted} files and ${res.embeddings_deleted} embeddings`);
            
            // Clear local state
            setTaskMap({});
            setProgressMap({});
            fetchFiles();
        } catch (error: any) {
            toast.dismiss();
            toast.error(error.message || 'Delete all failed');
        }
    };

    const handleReembedAll = async () => {
        if (!confirm(`Re-embed all ${files.length} files with ${selectedModel}? This will clear existing embeddings.`)) return;
        
        try {
            toast.loading('Starting re-embedding process...');
            const res = await fileApi.reembedAll(USER_ID, selectedModel);
            toast.dismiss();
            
            if (res.tasks) {
                // Set up task tracking for all files
                const newTaskMap: Record<string, string> = {};
                const newProgressMap: Record<string, number> = {};
                
                res.tasks.forEach((task: any) => {
                    newTaskMap[task.filename] = task.task_id;
                    newProgressMap[task.filename] = 0;
                });
                
                setTaskMap(prev => ({ ...prev, ...newTaskMap }));
                setProgressMap(prev => ({ ...prev, ...newProgressMap }));
            }
            
            toast.success(`Re-embedding ${res.count} files in background`);
            fetchFiles();
        } catch (error: any) {
            toast.dismiss();
            toast.error(error.message || 'Re-embed failed');
        }
    };

    // Filter logic needs to apply to the flattened list OR during the grouping
    // Simple approach: Search applies to the visual list
    // const filteredFiles = files.filter(f => f.filename.toLowerCase().includes(searchQuery.toLowerCase()));

    if (!isHydrated) return null;

    // Rendering helper for a file row
    const renderFileRow = (file: UploadedFile) => {
        if (searchQuery && !file.filename.toLowerCase().includes(searchQuery.toLowerCase())) return null;

        const progress = progressMap[file.filename] || 0;
        const isProcessing = (file.embedding_status === 'in_progress' || file.embedding_status === 'pending');
        
        // Use either real progress or indeterminate animation
        // If we have a task ID in map, we assume we have (or will have) progress
        // If not in map but status is pending, show indeterminate
        const showRealProgress = progressMap[file.filename] !== undefined;

        return (
            <div key={file.filename} className="bg-white p-6 rounded-xl border border-gray-200 hover:shadow-md transition flex items-center justify-between mb-4">
                <div className="flex-1">
                    <div className="flex items-center gap-3 mb-1">
                        <h3 className="text-lg font-bold text-gray-900 truncate max-w-xl" title={file.filename}>
                            {file.filename.split('/').pop()} {/* Show only Name part */}
                        </h3>
                        {file.embedding_status === 'completed' ? (
                            <span className="bg-green-100 text-green-800 text-xs font-bold px-2 py-0.5 rounded uppercase tracking-wide">Ready</span>
                        ) : (
                            <span className="bg-yellow-100 text-yellow-800 text-xs font-bold px-2 py-0.5 rounded uppercase tracking-wide">
                                {file.embedding_status === 'in_progress' ? 'Embedding' : 'Pending'}
                            </span>
                        )}
                    </div>
                    
                    <div className="text-xs text-gray-400 font-mono mb-2">/user_uploads/{USER_ID}/{file.filename}</div>

                    <div className="flex items-center gap-6 text-sm">
                        <span className="text-gray-600">{file.size ? (Number(file.size)/1024).toFixed(1) + ' KB' : 'Unknown Size'} • {file.embedding_status === 'completed' ? ((file as any).chunk_count || 'Unknown') + ' chunks' : ((file as any).chunk_count || 0) + ' chunks'}</span>
                        <span className="text-purple-600 text-xs bg-purple-50 px-2 py-1 rounded">
                            {file.embedding_model || 'nomic-embed-text-v1.5'}
                        </span>
                        <span className="text-gray-400 text-xs">{new Date(file.created_at || Date.now()).toLocaleString()}</span>
                    </div>
                </div>

                {isProcessing ? (
                    <div className="flex flex-col items-end gap-1 w-64 pl-6 border-l border-gray-100 ml-6">
                        <span className="text-xs font-medium text-yellow-600 uppercase tracking-wide bg-yellow-50 px-2 py-1 rounded mb-1">
                            Embedding
                        </span>
                        <div className="text-xs text-gray-500 mb-1">
                            Embedding {showRealProgress ? `${progress}%` : 'in progress...'}
                        </div>
                        <div className="w-full h-1 bg-gray-100 rounded-full overflow-hidden">
                             {showRealProgress ? (
                                <div 
                                    className="h-full bg-yellow-400 rounded-full transition-all duration-500 ease-out"
                                    style={{ width: `${Math.max(5, progress)}%` }} // Minimum width so it's visible
                                ></div>
                             ) : (
                                <div className="h-full bg-yellow-400 rounded-full animate-progress-indeterminate w-full origin-left-right"></div>
                             )}
                        </div>
                    </div>
                ) : (
                    <div className="flex items-center gap-4 pl-6 border-l border-gray-100 ml-6">
                        <button 
                            onClick={() => handleDelete(file.filename)}
                            className="text-red-500 hover:text-red-700 text-sm font-medium transition flex items-center gap-1"
                        >
                            <FiTrash2 /> Remove
                        </button>
                        <button 
                            onClick={() => handleRebuild(file.filename)}
                            className="text-green-600 hover:text-green-800 text-sm font-bold transition flex items-center gap-1 uppercase tracking-wide"
                        >
                            <FiRefreshCw /> Rebuild
                        </button>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="flex w-full h-screen bg-white">
            <Sidebar />
            <main className="flex-1 overflow-y-auto bg-gray-50 p-8">
                <style jsx>{`
                    @keyframes progress-indeterminate {
                        0% { transform: translateX(-100%); }
                        50% { transform: translateX(0%); }
                        100% { transform: translateX(100%); }
                    }
                    .animate-progress-indeterminate {
                        animation: progress-indeterminate 1.5s infinite linear;
                        width: 50%; /* Smaller bar moving across */
                    }
                `}</style>
                <Toaster />
                <div className="max-w-6xl mx-auto">
                    <header className="flex items-center justify-between mb-8">
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900">Local Documents</h1>
                            <p className="text-gray-500 text-sm mt-1">Manage your knowledge base for RAG</p>
                        </div>
                        <div className="flex gap-3">
                            {/* Embedding Model Selector */}
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="bg-white border border-gray-300 text-gray-700 text-sm rounded-lg focus:ring-green-500 focus:border-green-500 block px-3 py-2 shadow-sm"
                                title="Select Embedding Model"
                            >
                                {availableModels.map(model => (
                                    <option key={model.id} value={model.id}>
                                        {model.name}
                                    </option>
                                ))}
                            </select>

                            {files.length > 0 && (
                                <>
                                    <button
                                        onClick={handleDeleteAll}
                                        className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition flex items-center gap-2 font-medium shadow-sm"
                                        title="Delete all files and embeddings"
                                    >
                                        <FiTrash2 />
                                        Delete All
                                    </button>
                                    <button
                                        onClick={handleReembedAll}
                                        className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition flex items-center gap-2 font-medium shadow-sm"
                                        title="Re-embed all files with selected model"
                                    >
                                        <FiRefreshCw />
                                        Re-embed All
                                    </button>
                                </>
                            )}

                            <label className="bg-green-600 text-white px-4 py-2 rounded-lg cursor-pointer hover:bg-green-700 transition flex items-center gap-2 font-medium shadow-sm">
                                <FiFileText />
                                Add Files
                                <input 
                                    type="file" 
                                    className="hidden" 
                                    multiple
                                    accept=".pdf,.txt,.md"
                                    onChange={(e) => e.target.files && handleFileUpload(e.target.files)} 
                                />
                            </label>

                            <label className="bg-green-700 text-white px-4 py-2 rounded-lg cursor-pointer hover:bg-green-800 transition flex items-center gap-2 font-medium shadow-sm">
                                <FiFolder />
                                Add Folder
                                <input 
                                    type="file" 
                                    className="hidden" 
                                    {...{webkitdirectory: "", directory: ""} as any}
                                    onChange={(e) => e.target.files && handleFolderUpload(e.target.files)} 
                                />
                            </label>
                        </div>
                    </header>
                    
                    {/* Search & Stats */}
                    <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm mb-6 flex justify-between items-center">
                         <div className="relative flex-1 max-w-md">
                             <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                             <input 
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Search your files..." 
                                className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg outline-none focus:ring-2 focus:ring-green-500"
                             />
                         </div>
                         <div className="flex items-center gap-4 text-sm text-gray-500">
                             <span>{files.length} Files</span>
                             <span>{files.filter(f => f.available === 1).length} Ready</span>
                         </div>
                    </div>

                    {/* Progress Bar for Bulk Upload */}
                    {uploadProgress && (
                        <div className="mb-6 bg-white p-4 rounded-xl border border-blue-200 shadow-sm">
                            <div className="flex justify-between text-sm mb-2">
                                <span className="font-medium text-blue-800">{uploadProgress.message}</span>
                                <span className="text-blue-600">{uploadProgress.percentage}%</span>
                            </div>
                            <div className="w-full bg-blue-100 rounded-full h-2">
                                <div className="bg-blue-600 h-2 rounded-full transition-all duration-300" style={{ width: `${uploadProgress.percentage}%` }}></div>
                            </div>
                        </div>
                    )}

                    {/* File List grouped by folder */}
                    {files.length === 0 ? (
                        <div className="text-center py-20 text-gray-400">
                            <FiDatabase size={48} className="mx-auto mb-4 opacity-50" />
                            <p>No documents found.</p>
                        </div>
                    ) : (
                        <div className="flex flex-col space-y-4">
                            {/* Root Folder Files */}
                            {groupedFiles['root']?.map(file => renderFileRow(file))}

                            {/* Folders */}
                            {Object.entries(groupedFiles).map(([folderName, folderFiles]) => {
                                if (folderName === 'root') return null;
                                // Filter logic needs to check if any file in folder matches search
                                const matchesSearch = !searchQuery || folderFiles.some(f => f.filename.toLowerCase().includes(searchQuery.toLowerCase()));
                                if (!matchesSearch) return null;

                                const isExpanded = expandedFolders[folderName];
                                const readyCount = folderFiles.filter(f => f.embedding_status === 'completed').length;
                                
                                return (
                                    <div key={folderName} className="border border-gray-200 rounded-xl bg-white overflow-hidden mb-4">
                                        <div 
                                            className="px-6 py-4 flex items-center justify-between cursor-pointer bg-gray-50 hover:bg-gray-100 transition"
                                            onClick={() => toggleFolder(folderName)}
                                        >
                                            <div className="flex items-center gap-3">
                                                <FiFolder className={`text-xl ${readyCount === folderFiles.length ? 'text-green-500' : 'text-blue-500'}`} />
                                                <span className="font-bold text-gray-800 text-lg">{folderName}</span>
                                                <span className="text-gray-500 text-sm">({folderFiles.length} files)</span>
                                            </div>
                                            <div className="flex items-center gap-4">
                                                <div className="text-xs font-mono text-gray-400">
                                                    {readyCount}/{folderFiles.length} Ready
                                                </div>
                                                <svg 
                                                    className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
                                                    fill="none" viewBox="0 0 24 24" stroke="currentColor"
                                                >
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                                </svg>
                                            </div>
                                        </div>
                                        
                                        {isExpanded && (
                                            <div className="p-4 bg-white border-t border-gray-100 pl-8">
                                                {folderFiles.map(file => renderFileRow(file))}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}