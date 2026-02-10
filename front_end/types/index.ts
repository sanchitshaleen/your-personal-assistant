export interface User {
  user_id: string;
  email?: string;
  created_at?: string;
}

export interface UploadedFile {
  filename: string;
  created_at: string;
  size?: string;
  available: number;
  embedding_status?: 'pending' | 'in_progress' | 'completed' | 'failed';
  embedding_model?: string;
  error_message?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  files?: string[];
  timestamp: Date;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}

export interface ChatResponse {
  response: string;
  files_used?: string[];
  metadata?: Record<string, any>;
}

export interface ApiError {
  error: string;
  details?: string;
}
