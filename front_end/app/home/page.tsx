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

        {/* Main Content Area - Centered Welcome */}
        <div className="flex-1 flex flex-col items-center justify-center p-8 pb-32">
             <h1 className="text-5xl font-bold text-gray-900 mb-16 tracking-tight">SiloQ</h1>
             
             <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl w-full px-4">
                  {['Explain the project structure', 'Summarize latest logs', 'How do I add a new model?', 'Debug connection issues'].map((starter) => (
                       <button 
                          key={starter}
                          onClick={() => handleStarterClick(starter)}
                          className="p-6 bg-white border border-gray-200 rounded-2xl text-left text-gray-600 hover:border-green-500 hover:ring-1 hover:ring-green-500 hover:shadow-md transition duration-200 group"
                       >
                          <span className="group-hover:text-green-700 transition-colors">{starter}</span>
                       </button>
                  ))}
             </div>
        </div>
      </main>
    </div>
  );
}
