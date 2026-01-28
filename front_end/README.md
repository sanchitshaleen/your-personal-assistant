# SiloQ - Modern Frontend

A modern, clean Next.js/React frontend for SiloQ - Personal AI Data Assistant.

## Features

- 🔐 **Authentication**: Secure login/signup with support for Google OAuth
- 📁 **Dashboard**: Beautiful file management interface with multiple data source options
- 💬 **Chat Interface**: Real-time chat with context-aware responses
- 🎨 **Modern UI**: Clean, minimalist design with light beige and slate-gray color palette
- 📱 **Responsive**: Works seamlessly on desktop and mobile
- ⚡ **Fast**: Built with Next.js 14 and optimized for performance

## Tech Stack

- **Framework**: Next.js 14 (React 18)
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **HTTP Client**: Axios
- **Icons**: React Icons
- **Notifications**: React Hot Toast

## Project Structure

```
front_end/
├── app/
│   ├── page.tsx              # Auth page (login/signup)
│   ├── chat/
│   │   └── page.tsx          # Chat interface
│   ├── dashboard/
│   │   └── page.tsx          # File management dashboard
│   ├── layout.tsx            # Root layout
│   └── globals.css           # Global styles
├── lib/
│   ├── api.ts                # API client and endpoints
│   └── store.ts              # Zustand auth store
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── next.config.js
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- FastAPI backend running on `http://localhost:8002`

### Installation

```bash
cd front_end
npm install
```

### Environment Setup

```bash
cp .env.local.example .env.local
# Edit .env.local with your configuration
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## Screens

### 1. Authentication Page
- Modern login/signup interface
- Email and password fields
- Google OAuth support (coming soon)
- Social auth buttons
- Terms and privacy policy links

### 2. Dashboard
- **Left Sidebar**: Shows "Ready to Chat" files with sync status
- **Main Area**: Data source management
  - Manual file upload with drag-and-drop
  - Local laptop storage browser
  - Google Drive integration (coming soon)
- File management with clear/delete options

### 3. Chat Interface
- **Sidebar**: Conversation history
- **Main Chat Area**: Message history with timestamps
- **Context Selector**: Choose which files to use for responses
- **Input Area**: Send queries with file context

## API Integration

The frontend communicates with the FastAPI backend via:

- `POST /login` - User authentication
- `POST /signup` - User registration
- `GET /uploads` - Get user's uploaded files
- `POST /upload_file` - Upload new file
- `POST /clear_my_files` - Clear all uploads
- `POST /answer` - Send chat query
- `GET /answer_stream` - Stream chat responses

See `lib/api.ts` for all API endpoints and methods.

## Color Palette

- **Primary (Slate-gray)**: `#4A5568`
- **Secondary (Light beige)**: `#F5F3F0`
- **Accent (Green)**: `#48BB78`
- **Border**: `#E2E8F0`

## Future Enhancements

- [ ] Google Drive integration
- [ ] Multiple conversation management
- [ ] File sharing with other users
- [ ] Advanced search and filtering
- [ ] User settings and preferences
- [ ] Dark mode support
- [ ] Mobile app (React Native)

## Contributing

Please follow the project structure and coding conventions when contributing.

## License

MIT
