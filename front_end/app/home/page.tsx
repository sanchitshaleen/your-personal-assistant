'use client';

import Sidebar from '@/components/Sidebar';
import { FiChevronDown } from 'react-icons/fi';
import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

  const handleStarterClick = (starter: string) => {
    // In a real app, you might pass this starter text as a query param to /chat
    // e.g., router.push(`/chat?initialQuery=${encodeURIComponent(starter)}`);
    router.push('/chat'); 
  };

  return (
    <div className="flex w-full h-screen bg-white">
      <Sidebar />
      
      <main className="flex-1 flex flex-col h-screen relative bg-white">
        {/* Top Header - Consistent with Chat */}
        <header className="h-16 border-b border-gray-100 flex items-center justify-between px-6 bg-white z-10">
            <div className="flex items-center gap-2 text-sm text-gray-500 cursor-pointer hover:bg-gray-50 p-2 rounded-lg transition">
                <div className="w-5 h-5 rounded bg-green-600" /> 
                <span className="font-medium text-gray-700">Gemma-3</span>
                <FiChevronDown />
            </div>
            
            {/* Can add user profile or other header items here if needed */}
        </header>

        {/* Main Content Area - App Description */}
        <div className="flex-1 flex flex-col items-center justify-center p-8 pb-32">
          <h1 className="text-5xl font-bold text-gray-900 mb-8 tracking-tight">SiloQ</h1>
          <p className="max-w-2xl text-lg text-gray-700 mb-12 text-center">
            <b>SiloQ</b> is your personal AI-powered assistant for chat, document Q&A, and model management. Start a new chat, explore available models, or upload your own documents for retrieval-augmented generation. SiloQ is designed to help you interact with your data and AI models in a seamless, user-friendly way.
          </p>
          {/* Sample questions removed as requested */}
        </div>
      </main>
    </div>
  );
}
