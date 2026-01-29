'use client';

import { useState, useEffect } from 'react';
import { fileApi } from '@/lib/api';
import Sidebar from '@/components/Sidebar';
import { FiBox, FiCpu, FiDownload, FiTrash2, FiCheck, FiDatabase, FiArrowLeft } from 'react-icons/fi';
import toast, { Toaster } from 'react-hot-toast';

interface Model {
    id: string;
    name: string;
    type: 'llm' | 'embedding';
    isActive?: boolean;
    size?: number;
    details?: any;
}

interface RecommendedModel {
    name: string;
    displayName: string;
    description: string;
    size: string;
    parameters: string;
    type: 'llm' | 'embedding';
    recommended?: boolean;
}

const RECOMMENDED_LLM_MODELS: RecommendedModel[] = [
    // Lightweight & Fast
    {
        name: 'gemma3:270m',
        displayName: 'Gemma 3 270M',
        description: 'Ultra-lightweight for rapid responses on low-resource devices',
        size: '292 MB',
        parameters: '270M',
        type: 'llm',
        recommended: true
    },
    {
        name: 'gemma3:1b',
        displayName: 'Gemma 3 1B',
        description: 'Fast, lightweight model ideal for quick responses',
        size: '815 MB',
        parameters: '1B',
        type: 'llm',
        recommended: true
    },
    {
        name: 'gemma3:2b',
        displayName: 'Gemma 3 2B',
        description: 'Balanced performance and quality for general tasks',
        size: '1.6 GB',
        parameters: '2B',
        type: 'llm'
    },
    {
        name: 'llama3.2:3b',
        displayName: 'Llama 3.2 3B',
        description: 'Fast responses with good quality, optimized for chat',
        size: '2.0 GB',
        parameters: '3B',
        type: 'llm'
    },
    // Medium Performance
    {
        name: 'mistral:7b',
        displayName: 'Mistral 7B',
        description: 'High quality responses, excellent for complex reasoning',
        size: '4.1 GB',
        parameters: '7B',
        type: 'llm'
    },
    {
        name: 'llama3:8b',
        displayName: 'Llama 3 8B',
        description: 'Meta\'s powerful model with exceptional performance',
        size: '4.7 GB',
        parameters: '8B',
        type: 'llm'
    },
    {
        name: 'granite4:3b',
        displayName: 'Granite 4 3B',
        description: 'IBM\'s enterprise model with improved instruction following',
        size: '2.0 GB',
        parameters: '3B',
        type: 'llm'
    },
    // Advanced Models
    {
        name: 'qwen3-next:80b',
        displayName: 'Qwen 3 Next 80B',
        description: 'Strong parameter efficiency with advanced reasoning',
        size: '48 GB',
        parameters: '80B',
        type: 'llm'
    },
    {
        name: 'deepseek-v3.1',
        displayName: 'DeepSeek V3.1',
        description: 'Hybrid model with thinking and non-thinking modes',
        size: '400+ GB',
        parameters: '671B',
        type: 'llm'
    }
];

const RECOMMENDED_EMBEDDING_MODELS: RecommendedModel[] = [
    // Popular & Recommended
    {
        name: 'nomic-embed-text',
        displayName: 'Nomic Embed Text',
        description: 'High-performing with large context window (52M+ pulls)',
        size: '274 MB',
        parameters: '137M',
        type: 'embedding',
        recommended: true
    },
    {
        name: 'mxbai-embed-large',
        displayName: 'MXBai Embed Large',
        description: 'State-of-the-art embeddings from mixedbread.ai (7M+ pulls)',
        size: '669 MB',
        parameters: '335M',
        type: 'embedding',
        recommended: true
    },
    // Multilingual
    {
        name: 'bge-m3',
        displayName: 'BGE-M3',
        description: 'Versatile multilingual model from BAAI (3M+ pulls)',
        size: '600 MB',
        parameters: '567M',
        type: 'embedding'
    },
    {
        name: 'paraphrase-multilingual',
        displayName: 'Paraphrase Multilingual',
        description: 'Excellent for clustering and semantic search across languages',
        size: '300 MB',
        parameters: '278M',
        type: 'embedding'
    },
    // Specialized
    {
        name: 'snowflake-arctic-embed',
        displayName: 'Snowflake Arctic Embed',
        description: 'Optimized for performance with multiple size options (2M+ pulls)',
        size: '137-669 MB',
        parameters: '137M-335M',
        type: 'embedding'
    },
    {
        name: 'all-minilm',
        displayName: 'All-MiniLM',
        description: 'Compact and fast for sentence-level embeddings (2M+ pulls)',
        size: '23-33 MB',
        parameters: '22M-33M',
        type: 'embedding'
    },
    {
        name: 'bge-large',
        displayName: 'BGE Large',
        description: 'High-quality embeddings from BAAI for text-to-vector mapping',
        size: '340 MB',
        parameters: '335M',
        type: 'embedding'
    },
    {
        name: 'granite-embedding',
        displayName: 'Granite Embedding',
        description: 'IBM\'s dense biencoder with English and multilingual variants',
        size: '30-280 MB',
        parameters: '30M-278M',
        type: 'embedding'
    },
    // Latest Advanced
    {
        name: 'qwen3-embedding',
        displayName: 'Qwen 3 Embedding',
        description: 'Comprehensive range of sizes from Qwen3 series (400K+ pulls)',
        size: '600MB-8GB',
        parameters: '0.6B-8B',
        type: 'embedding'
    },
    {
        name: 'embeddinggemma',
        displayName: 'Embedding Gemma',
        description: 'Google\'s 300M parameter embedding model (452K+ pulls)',
        size: '300 MB',
        parameters: '300M',
        type: 'embedding'
    },
    {
        name: 'snowflake-arctic-embed2',
        displayName: 'Snowflake Arctic Embed 2',
        description: 'Frontier embedding with multilingual support',
        size: '570 MB',
        parameters: '568M',
        type: 'embedding'
    }
];

export default function ModelsPage() {
    const [models, setModels] = useState<Model[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [pullModelName, setPullModelName] = useState('');
    const [isPulling, setIsPulling] = useState(false);
    const [showExplorer, setShowExplorer] = useState(false);
    const [currentTab, setCurrentTab] = useState<'llm' | 'embedding'>('llm');
    const [selectedLlmModel, setSelectedLlmModel] = useState('');
    const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState('');

    const fetchModels = async () => {
        try {
            const res = await fileApi.getModels();
            const activeMap = res.active || {};
            let finalModels: Model[] = [];
            
            if (res.models && res.models.length > 0) {
                finalModels = res.models.map((m: any) => ({
                    ...m,
                    isActive: m.isActive || (m.type === 'llm' && activeMap.llm === m.name) || (m.type === 'embedding' && activeMap.embedding === m.name)
                }));
            } else {
                // If no models from Ollama, create entries from active config
                if (activeMap.llm) {
                    finalModels.push({
                        id: activeMap.llm,
                        name: activeMap.llm,
                        type: 'llm',
                        isActive: true,
                        size: undefined
                    });
                }
                if (activeMap.embedding) {
                    finalModels.push({
                        id: activeMap.embedding,
                        name: activeMap.embedding,
                        type: 'embedding',
                        isActive: true,
                        size: undefined
                    });
                }
            }
            
            setModels(finalModels);
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
    
    const handleDownloadFromDropdown = async (modelName: string, type: 'llm' | 'embedding') => {
        if (!modelName) return;
        
        setIsPulling(true);
        try {
            toast.loading(`Downloading ${modelName}... This may take several minutes`);
            await fileApi.pullModel(modelName);
            toast.dismiss();
            toast.success(`${modelName} downloaded successfully!`);
            
            // Reset the dropdown
            if (type === 'llm') {
                setSelectedLlmModel('');
            } else {
                setSelectedEmbeddingModel('');
            }
            
            setTimeout(fetchModels, 2000);
        } catch (error: any) {
            toast.dismiss();
            toast.error(error.message || 'Download failed');
        } finally {
            setIsPulling(false);
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
    
    const handleDownloadRecommended = async (modelName: string) => {
        setIsPulling(true);
        try {
            toast.loading(`Downloading ${modelName}... This may take a few minutes`);
            await fileApi.pullModel(modelName);
            toast.dismiss();
            toast.success(`${modelName} downloaded successfully!`);
            setTimeout(() => {
                fetchModels();
                setShowExplorer(false);
            }, 2000);
        } catch (error: any) {
            toast.dismiss();
            toast.error(error.message || 'Download failed');
        } finally {
            setIsPulling(false);
        }
    };
    
    const renderRecommendedCard = (model: RecommendedModel) => {
        // Check if model is installed (handle tag variants like :latest)
        const isInstalled = models.some(m => {
            const modelBaseName = m.name.split(':')[0];
            const recommendedBaseName = model.name.split(':')[0];
            return m.name === model.name || modelBaseName === recommendedBaseName;
        });
        
        // Get the actual installed model name if it exists
        const installedModel = models.find(m => {
            const modelBaseName = m.name.split(':')[0];
            const recommendedBaseName = model.name.split(':')[0];
            return m.name === model.name || modelBaseName === recommendedBaseName;
        });
        
        return (
            <div key={model.name} className="bg-white border border-gray-200 rounded-xl p-6 hover:border-green-300 hover:shadow-md transition relative">
                {model.recommended && (
                    <div className="absolute top-4 right-4 bg-green-100 text-green-800 text-xs font-bold px-2 py-1 rounded-full">
                        Recommended
                    </div>
                )}
                
                <h3 className="font-bold text-lg text-gray-900 mb-2">{model.displayName}</h3>
                <p className="text-sm text-gray-600 mb-4 leading-relaxed">{model.description}</p>
                
                <div className="flex gap-2 mb-6">
                    <span className="text-xs bg-gray-100 text-gray-600 px-3 py-1 rounded-full border border-gray-200">
                        📦 {model.size}
                    </span>
                    <span className="text-xs bg-gray-100 text-gray-600 px-3 py-1 rounded-full border border-gray-200">
                        🧠 {model.parameters}
                    </span>
                </div>
                
                {isInstalled ? (
                    <div className="flex gap-2">
                        <div className="flex-1 py-2 bg-green-50 text-green-700 rounded-lg font-medium text-sm flex items-center justify-center gap-2">
                            <FiCheck /> Installed
                        </div>
                        {installedModel && !installedModel.isActive && (
                            <button
                                onClick={() => handleDelete(installedModel.name)}
                                className="px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition flex items-center gap-2 text-sm font-medium"
                                title="Uninstall model to free up space"
                            >
                                <FiTrash2 />
                            </button>
                        )}
                    </div>
                ) : (
                    <button
                        onClick={() => handleDownloadRecommended(model.name)}
                        disabled={isPulling}
                        className="w-full py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white rounded-lg font-medium text-sm flex items-center justify-center gap-2 transition"
                    >
                        <FiDownload /> Download
                    </button>
                )}
            </div>
        );
    };

  return (
    <div className="flex w-full h-screen bg-gray-50">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-8">
        <Toaster />
        <div className="max-w-6xl mx-auto">
          {showExplorer ? (
              /* Model Explorer View */
              <>
                  <div className="mb-8 flex items-center gap-4">
                      <button
                          onClick={() => setShowExplorer(false)}
                          className="p-2 hover:bg-gray-200 rounded-lg transition"
                      >
                          <FiArrowLeft className="text-gray-600" />
                      </button>
                      <div>
                          <h1 className="text-2xl font-bold text-gray-900">Explore Models</h1>
                          <p className="text-gray-500 text-sm mt-1">
                              These models have been specifically configured for Your Personal Assistant
                          </p>
                      </div>
                  </div>
                  
                  {/* Tabs */}
                  <div className="flex gap-4 mb-6 border-b border-gray-200">
                      <button
                          onClick={() => setCurrentTab('llm')}
                          className={`px-4 py-2 font-medium transition border-b-2 ${
                              currentTab === 'llm'
                                  ? 'border-green-600 text-green-600'
                                  : 'border-transparent text-gray-500 hover:text-gray-700'
                          }`}
                      >
                          LLM Models
                      </button>
                      <button
                          onClick={() => setCurrentTab('embedding')}
                          className={`px-4 py-2 font-medium transition border-b-2 ${
                              currentTab === 'embedding'
                                  ? 'border-green-600 text-green-600'
                                  : 'border-transparent text-gray-500 hover:text-gray-700'
                          }`}
                      >
                          Embedding Models
                      </button>
                  </div>
                  
                  {/* Model Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {currentTab === 'llm'
                          ? RECOMMENDED_LLM_MODELS.map(renderRecommendedCard)
                          : RECOMMENDED_EMBEDDING_MODELS.map(renderRecommendedCard)}
                  </div>
              </>
          ) : (
              /* Installed Models View */
              <>
                  <header className="mb-8">
                    <h1 className="text-2xl font-bold text-gray-900">Models</h1>
                    <p className="text-gray-500 text-sm mt-1">Manage and select AI models for your workspace.</p>
                  </header>

          {/* Quick Download Section */}
          <div className="mb-8 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-xl p-6">
              <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <FiDownload className="text-green-600" /> Quick Download
              </h2>
              <p className="text-sm text-gray-600 mb-4">
                  Select a model from the dropdown and click download to install it
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* LLM Model Dropdown */}
                  <div className="bg-white rounded-lg p-4 border border-gray-200">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                          Response Generation Model (LLM)
                      </label>
                      <div className="flex gap-2">
                          <select
                              value={selectedLlmModel}
                              onChange={(e) => setSelectedLlmModel(e.target.value)}
                              disabled={isPulling}
                              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-gray-100 text-sm"
                          >
                              <option value="">Select an LLM model...</option>
                              {RECOMMENDED_LLM_MODELS.map((model) => {
                                  const isInstalled = models.some(m => {
                                      const modelBaseName = m.name.split(':')[0];
                                      const recommendedBaseName = model.name.split(':')[0];
                                      return m.name === model.name || modelBaseName === recommendedBaseName;
                                  });
                                  return (
                                      <option key={model.name} value={model.name} disabled={isInstalled}>
                                          {model.displayName} ({model.parameters}) {isInstalled ? '✓ Installed' : ''}
                                      </option>
                                  );
                              })}
                          </select>
                          <button
                              onClick={() => handleDownloadFromDropdown(selectedLlmModel, 'llm')}
                              disabled={!selectedLlmModel || isPulling}
                              className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white rounded-lg font-medium text-sm transition flex items-center gap-2 whitespace-nowrap"
                          >
                              <FiDownload /> Download
                          </button>
                      </div>
                  </div>
                  
                  {/* Embedding Model Dropdown */}
                  <div className="bg-white rounded-lg p-4 border border-gray-200">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                          Embedding Model
                      </label>
                      <div className="flex gap-2">
                          <select
                              value={selectedEmbeddingModel}
                              onChange={(e) => setSelectedEmbeddingModel(e.target.value)}
                              disabled={isPulling}
                              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-gray-100 text-sm"
                          >
                              <option value="">Select an embedding model...</option>
                              {RECOMMENDED_EMBEDDING_MODELS.map((model) => {
                                  const isInstalled = models.some(m => {
                                      const modelBaseName = m.name.split(':')[0];
                                      const recommendedBaseName = model.name.split(':')[0];
                                      return m.name === model.name || modelBaseName === recommendedBaseName;
                                  });
                                  return (
                                      <option key={model.name} value={model.name} disabled={isInstalled}>
                                          {model.displayName} ({model.parameters}) {isInstalled ? '✓ Installed' : ''}
                                      </option>
                                  );
                              })}
                          </select>
                          <button
                              onClick={() => handleDownloadFromDropdown(selectedEmbeddingModel, 'embedding')}
                              disabled={!selectedEmbeddingModel || isPulling}
                              className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white rounded-lg font-medium text-sm transition flex items-center gap-2 whitespace-nowrap"
                          >
                              <FiDownload /> Download
                          </button>
                      </div>
                  </div>
              </div>
              
              <div className="mt-4 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span className="font-medium">💡 Tip:</span>
                      <span>Or browse all available models below</span>
                  </div>
                  <button
                      onClick={() => setShowExplorer(true)}
                      className="px-4 py-2 bg-white border border-green-600 text-green-600 rounded-lg hover:bg-green-50 transition flex items-center gap-2 text-sm font-medium"
                  >
                      <FiBox /> Browse All Models
                  </button>
              </div>
          </div>

          {/* Generation Models */}
          <div className="mb-10">
            <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                <FiCpu /> Response Generation Models (LLMs)
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {llmModels.map(renderModelCard)}
                
                {llmModels.length === 0 && !isLoading && (
                    <div className="col-span-full p-8 border-2 border-dashed border-gray-300 rounded-xl flex flex-col items-center justify-center text-gray-400 py-12 bg-white">
                        <FiBox size={48} className="mb-4 opacity-50" />
                        <p className="text-gray-500 mb-4">No LLM models installed</p>
                        <button
                            onClick={() => setShowExplorer(true)}
                            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition flex items-center gap-2"
                        >
                            <FiDownload /> Browse Models
                        </button>
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
                    <div className="col-span-full p-8 border-2 border-dashed border-gray-300 rounded-xl flex flex-col items-center justify-center text-gray-400 py-12 bg-white">
                        <FiBox size={48} className="mb-4 opacity-50" />
                        <p className="text-gray-500 mb-4">No Embedding models installed</p>
                        <button
                            onClick={() => {
                                setCurrentTab('embedding');
                                setShowExplorer(true);
                            }}
                            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition flex items-center gap-2"
                        >
                            <FiDownload /> Browse Models
                        </button>
                    </div>
                )}
            </div>
          </div>
          </>
          )}
        </div>
      </main>
    </div>
  );
}
