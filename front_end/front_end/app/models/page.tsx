'use client';

import { useState, useEffect } from 'react';
import { fileApi } from '@/lib/api'; // Using fileApi for getModels as added previously
import Sidebar from '@/components/Sidebar';
import { FiBox, FiCpu, FiDownload, FiTrash2, FiActivity, FiCheck, FiDatabase } from 'react-icons/fi';
import toast, { Toaster } from 'react-hot-toast';

interface Model {
    id: string;
    name: string;
    type: 'llm' | 'embedding';
    isActive?: boolean;
    size?: number;
    details?: any;
}

export default function ModelsPage() {
    const [models, setModels] = useState<Model[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [pullModelName, setPullModelName] = useState('');
    const [isPulling, setIsPulling] = useState(false);

    const fetchModels = async () => {
        try {
            const res = await fileApi.getModels();
            // Expecting { models: Model[], active: { llm: string, embedding: string }}
            if (res.models) {
                const activeMap = res.active || {};
                
                const mappedModels = res.models.map((m: any) => ({
                    ...m,
                    isActive: m.isActive || (m.type === 'llm' && activeMap.llm === m.name) || (m.type === 'embedding' && activeMap.embedding === m.name)
                }));
                
                setModels(mappedModels);
            }
        } catch (error) {
            console.error('Failed to fetch models', error);
            toast.error('Failed to load models');
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchModels();
    }, []);

    const handlePull = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!pullModelName) return;

        setIsPulling(true);
        try {
            toast.loading(`Starting pull for ${pullModelName}... check terminal for progress`);
            await fileApi.pullModel(pullModelName);
            toast.dismiss();
            toast.success(`Pull request for ${pullModelName} sent. It will appear when ready.`);
            setPullModelName('');
            setTimeout(fetchModels, 5000); 
        } catch (error: any) {
             toast.dismiss();
             toast.error(error.message || 'Pull failed');
        } finally {
            setIsPulling(false);
        }
    };

    const handleDelete = async (name: string) => {
        if (!confirm(`Are you sure you want to delete ${name}?`)) return;
        try {
            await fileApi.deleteModel(name);
            toast.success('Model deleted');
            fetchModels();
        } catch (error: any) {
            toast.error(error.message || 'Delete failed');
        }
    };

    const handleSetActive = async (model: Model) => {
        try {
            await fileApi.setActiveModel(model.type, model.name);
            toast.success(`Set ${model.name} as active ${model.type} model`);
            fetchModels(); 
        } catch (error: any) {
            toast.error('Failed to set active model');
        }
    };

    const formatSize = (bytes?: number) => {
        if (!bytes) return 'Unknown size';
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    };

    const renderModelCard = (model: Model) => (
        <div key={model.id} className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm hover:shadow-md transition relative overflow-hidden">
            {model.isActive && (
                <div className="absolute top-4 right-4 bg-green-100 text-green-800 text-xs font-bold px-2 py-1 rounded-full flex items-center gap-1">
                    <FiCheck /> Active
                </div>
            )}
            
            <div className="flex items-start justify-between mb-4">
                <div className={`p-3 rounded-lg ${model.type === 'llm' ? 'bg-blue-50' : 'bg-purple-50'}`}>
                    <FiBox className={`${model.type === 'llm' ? 'text-blue-600' : 'text-purple-600'} text-xl`} />
                </div>
                {!model.isActive && (
                   <button 
                       onClick={() => handleDelete(model.name)}
                       className="text-gray-400 hover:text-red-500 transition"
                       title="Delete Model"
                   >
                       <FiTrash2 />
                   </button>
                )}
            </div>

            <h3 className="font-bold text-lg text-gray-900 mb-1">{model.name}</h3>
            
            <div className="flex gap-2 mb-4">
                 <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded border border-gray-200">
                    {formatSize(model.size)}
                 </span>
                 {model.details?.parameter_size && (
                     <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded border border-gray-200">
                        {model.details.parameter_size}
                     </span>
                 )}
                {model.details?.quantization_level && (
                     <span className="text-xs bg-gray-50 text-gray-500 px-2 py-1 rounded border border-gray-100">
                        {model.details.quantization_level}
                     </span>
                 )}
            </div>

            <div className="text-sm text-gray-500 mb-6 font-mono">
                {model.type === 'llm' ? 'LLM' : 'Embedding'}
            </div>

            {!model.isActive && (
                <button 
                    onClick={() => handleSetActive(model)}
                    className="w-full py-2 border border-gray-200 hover:border-green-500 hover:text-green-600 text-gray-600 rounded-lg font-medium transition text-sm flex items-center justify-center gap-2"
                >
                    Set Active
                </button>
            )}
            {model.isActive && (
                 <div className="w-full py-2 bg-green-50 text-green-700 rounded-lg font-medium text-sm flex items-center justify-center gap-2 cursor-default">
                    Currently Active
                </div>
            )}
        </div>
    );

    const llmModels = models.filter(m => m.type === 'llm');
    const embedModels = models.filter(m => m.type === 'embedding');

    return (
        <div className="flex w-full h-screen bg-gray-50">
            <Sidebar />
            <main className="flex-1 overflow-y-auto p-8">
                <Toaster />
                <div className="max-w-6xl mx-auto">
                    <header className="mb-8">
                        <h1 className="text-2xl font-bold text-gray-900">Models</h1>
                        <p className="text-gray-500 text-sm mt-1">Manage and select AI models for your workspace.</p>
                    </header>

                    {/* Generation Models */}
                    <div className="mb-10">
                        <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                            <FiCpu /> Response Generation Models (LLMs)
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {llmModels.map(renderModelCard)}
                            
                            {/* Empty State / Add New Card for LLMs */}
                            {llmModels.length === 0 && !isLoading && (
                                <div className="col-span-1 p-6 border border-dashed border-gray-300 rounded-xl flex flex-col items-center justify-center text-gray-400 py-12">
                                    <FiBox size={32} className="mb-2 opacity-50" />
                                    <span>No LLM models installed</span>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Embedding Models */}
                    <div className="mb-10">
                        <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                             <FiDatabase /> Embedding Models
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {embedModels.map(renderModelCard)}
                             {embedModels.length === 0 && !isLoading && (
                                <div className="col-span-1 p-6 border border-dashed border-gray-300 rounded-xl flex flex-col items-center justify-center text-gray-400 py-12">
                                    <FiBox size={32} className="mb-2 opacity-50" />
                                    <span>No Embedding models installed</span>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Add New Model Section */}
                    <div className="bg-white p-8 rounded-xl border border-gray-200 shadow-sm border-dashed border-2 border-green-200">
                        <div className="flex flex-col items-center justify-center text-center max-w-lg mx-auto">
                            <div className="w-12 h-12 bg-green-50 rounded-full flex items-center justify-center mb-4 text-green-600">
                                <FiDownload size={24} />
                            </div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">Add New Model</h3>
                            <p className="text-gray-500 mb-6">
                                Download generic models directly from the Ollama library.
                                (e.g. "llama3", "mistral", "gemma:2b")
                            </p>
                            
                            <form onSubmit={handlePull} className="flex gap-2 w-full">
                                <input 
                                    type="text" 
                                    placeholder="Enter model tag (e.g. mistral)" 
                                    className="flex-1 px-4 py-3 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-green-500"
                                    value={pullModelName}
                                    onChange={(e) => setPullModelName(e.target.value)}
                                    disabled={isPulling}
                                />
                                <button 
                                    type="submit"
                                    disabled={isPulling || !pullModelName}
                                    className={`px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 transition ${isPulling ? 'opacity-50 cursor-not-allowed' : ''}`}
                                >
                                    {isPulling ? 'Pulling...' : 'Download'}
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
