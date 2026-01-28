'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FiHome, FiMessageSquare, FiBox, FiDatabase } from 'react-icons/fi';

export default function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    { name: 'Chats', icon: FiMessageSquare, href: '/chat' },
    { name: 'Models', icon: FiBox, href: '/models' },
    { name: 'Local Documents', icon: FiDatabase, href: '/dashboard' },
  ];

  return (
    <div className="w-64 bg-gray-50 border-r border-gray-200 h-screen flex flex-col flex-shrink-0">
      {/* Branding/Logo Area */}
      <div className="p-6 flex items-center justify-center border-b border-gray-100">
         <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
            <FiHome className="text-green-700 w-6 h-6" />
         </div>
         <span className="ml-3 font-bold text-xl text-green-900">SiloQ</span>
      </div>

      <div className="p-4">
          <Link href="/chat" className="block w-full bg-green-50 hover:bg-green-100 text-green-800 font-medium py-3 px-4 rounded-lg text-center transition mb-6 border border-green-200 shadow-sm">
            + New Chat
          </Link>

          <div className="space-y-1">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition ${
                    isActive
                      ? 'bg-green-100 text-green-900'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  <item.icon size={18} />
                  {item.name}
                </Link>
              );
            })}
          </div>
      </div>
    </div>
  );
}
