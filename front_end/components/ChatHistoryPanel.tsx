'use client';

import React from 'react';

interface Props {
  conversations: any[];
  onSelect: (conv: any) => void;
  onClose: () => void;
  onNew?: () => void;
  onDelete?: (id: string) => void;
}

export default function ChatHistoryPanel({ conversations, onSelect, onClose, onNew, onDelete }: Props) {
  return (
    <div className="h-full flex flex-col bg-gray-50 border-l border-gray-100">
      <div className="p-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Conversation History</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600">Close</button>
      </div>

      <div className="px-4 pb-4">
        <button onClick={() => onNew && onNew()} className="w-full p-3 bg-green-50 rounded-lg text-sm font-medium text-green-800 text-left">+ New Chat</button>
      </div>

      <div className="px-4 text-xs text-gray-400 uppercase tracking-wider">Today</div>

      <div className="p-4 space-y-3 overflow-y-auto flex-1">
        {conversations.length === 0 ? (
          <div className="text-xs text-gray-400">No previous chats.</div>
        ) : (
          conversations.map((c) => (
            <div key={c.id} className="relative p-3 bg-white border border-gray-100 rounded-lg hover:shadow">
              <div className="flex-1 cursor-pointer pr-10" onClick={() => onSelect(c)}>
                <div className="text-sm font-medium text-gray-800 truncate">{c.title}</div>
                <div className="text-xs text-gray-500 mt-1">{new Date(c.createdAt).toLocaleString()}</div>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); onDelete && onDelete(c.id); }}
                className="absolute top-2 right-2 text-gray-400 hover:text-red-600 p-2 rounded"
                title="Delete conversation"
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
