'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/store';

export default function AuthPage() {
  const router = useRouter();
  const { user, loginAsGuest } = useAuthStore();

  useEffect(() => {
    // Automatically log in as guest if not logged in
    if (!user) {
      loginAsGuest();
    } else {
      router.push('/home');
    }
  }, [user, router, loginAsGuest]);

  return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <div className="text-center">
        <div className="mb-4">
          <div className="w-16 h-16 bg-black rounded-lg flex items-center justify-center mx-auto mb-4">
            <span className="text-white font-bold text-2xl">S</span>
          </div>
          <h2 className="text-2xl font-bold text-gray-900">Welcome to SiloQ</h2>
          <p className="text-gray-600 mt-2">Entering your private workspace...</p>
        </div>
      </div>
    </div>
  );
}
