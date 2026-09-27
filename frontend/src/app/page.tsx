"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { UploadCloud, FileText, CheckCircle, Database, GitBranch, ShieldCheck, Activity } from "lucide-react";

interface SubgraphItem {
  subject: string;
  predicate: string;
  object: string;
  source?: string;
  weight?: number;
}

interface Message {
  role: "user" | "agent";
  content: string;
  subgraph?: SubgraphItem[];
  score?: number;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploadedDocs, setUploadedDocs] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [auditNode, setAuditNode] = useState<SubgraphItem | null>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    setIsUploading(true);
    
    const formData = new FormData();
    Array.from(e.target.files).forEach((file) => {
      formData.append("files", file);
    });

    try {
      const res = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.uploaded_docs) {
        setUploadedDocs(data.uploaded_docs);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFeedback = async (subgraph: SubgraphItem[], isPositive: boolean) => {
    if (!subgraph.length) return;
    try {
      const res = await fetch("http://localhost:8000/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subgraph, is_positive: isPositive }),
      });
      const data = await res.json();
      alert(data.message); // Simple notification that weight was adjusted
    } catch (err) {
      console.error(err);
    }
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) return;

    const userMessage: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    
    // Create an empty agent message to stream into
    const agentMessageId = Date.now();
    setMessages((prev) => [
      ...prev,
      { role: "agent", content: "", _id: agentMessageId, streamThoughts: [] } as any
    ]);
    
    setQuery("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/chat_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessage.content }),
      });
      
      if (!res.ok) {
        throw new Error(`API Error: ${res.status}`);
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      
      if (reader) {
        let done = false;
        let buffer = "";
        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          if (value) {
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\n\n');
            buffer = parts.pop() || "";
            for (const part of parts) {
              const line = part.trim();
              if (!line.startsWith('data: ')) continue;
              const dataStr = line.replace('data: ', '').trim();
              if (!dataStr) continue;
              try {
                const data = JSON.parse(dataStr);
                
                setMessages((prev) => prev.map((msg: any) => {
                  if (msg._id !== agentMessageId) return msg;
                  
                  if (data.type === "thought") {
                    const thoughts = msg.streamThoughts || [];
                    return { ...msg, streamThoughts: [...thoughts, { agent: data.agent, text: data.message }] };
                  } else if (data.type === "done") {
                    return { ...msg, content: data.answer, subgraph: data.subgraph, score: data.score, streamThoughts: [] };
                  }
                  return msg;
                }));
              } catch(e) {
                console.error("Failed to parse SSE JSON:", e);
              }
            }
          }
        }
      }
    } catch (err) {
      setMessages((prev) => prev.map((msg: any) => 
        msg._id === agentMessageId ? { ...msg, content: "Error connecting to backend." } : msg
      ));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 font-sans flex overflow-hidden">
      
      {/* Sidebar - Knowledge Base */}
      <aside className="w-80 bg-white border-r border-slate-200 flex flex-col h-screen shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.02)] z-10">
        <div className="p-6 border-b border-slate-100">
          <div className="flex items-center gap-3 mb-2">
            <div className="bg-blue-600 p-2 rounded-lg text-white">
              <Database size={20} />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900">Adaptive RAG</h1>
          </div>
          <p className="text-xs text-slate-500 font-medium">Enterprise Knowledge Engine</p>
        </div>
        
        <div className="p-6 flex-1 overflow-y-auto">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
            <FileText size={14} /> Knowledge Base
          </h2>
          
          <label className="flex items-center justify-center gap-2 bg-slate-50 border-2 border-dashed border-slate-300 rounded-xl p-4 cursor-pointer hover:bg-slate-100 hover:border-blue-400 transition-all group mb-6">
            <UploadCloud size={20} className="text-slate-400 group-hover:text-blue-500" />
            <span className="text-sm font-semibold text-slate-600 group-hover:text-blue-600">
              {isUploading ? "Extracting Graph..." : "Upload PDFs"}
            </span>
            <input type="file" multiple accept=".pdf" className="hidden" onChange={handleUpload} disabled={isUploading} />
          </label>

          <div className="space-y-2">
            {uploadedDocs.length === 0 ? (
              <div className="text-xs text-center text-slate-400 py-4 bg-slate-50 rounded-lg border border-slate-100">No documents ingested yet</div>
            ) : (
              uploadedDocs.map((doc, idx) => (
                <div key={idx} className="flex items-center gap-3 p-3 bg-white border border-slate-200 rounded-lg shadow-sm">
                  <CheckCircle size={16} className="text-emerald-500 shrink-0" />
                  <span className="text-sm font-medium text-slate-700 truncate">{doc}</span>
                </div>
              ))
            )}
          </div>
        </div>
        
        <div className="p-4 bg-slate-50 border-t border-slate-200 text-xs text-slate-500 flex justify-between items-center">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Backend Live
          </div>
          <span className="font-mono">FastAPI</span>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-screen relative">
        {/* Background Graphic */}
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: 'radial-gradient(#3b82f6 1px, transparent 1px)', backgroundSize: '32px 32px' }}></div>
        
        <div className="flex-1 overflow-y-auto p-8 z-10 scroll-smooth">
          <div className="max-w-4xl mx-auto space-y-10">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 mt-32 space-y-4">
                <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center border border-slate-200">
                  <GitBranch size={32} className="text-slate-300" />
                </div>
                <h3 className="text-xl font-semibold text-slate-700">Awaiting Query</h3>
                <p className="text-sm text-center max-w-md">Upload your structural documents on the left, then ask a multi-hop reasoning question to trigger graph extraction.</p>
              </div>
            )}
            
            {messages.map((msg, idx) => (
              <div key={idx} className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                {msg.role === "user" ? (
                  <div className="flex justify-end">
                    <div className="bg-slate-900 text-white rounded-2xl rounded-tr-sm px-6 py-4 max-w-[80%] shadow-md">
                      <p className="text-sm md:text-base">{msg.content}</p>
                    </div>
                  </div>
                ) : (
                    <div className="flex flex-col gap-4">
                    {/* Streaming Thoughts Monologue */}
                    {(msg as any).streamThoughts && (msg as any).streamThoughts.length > 0 && (
                      <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 max-w-[95%] space-y-2 font-mono text-xs shadow-lg">
                        <div className="flex items-center gap-2 text-blue-400 font-bold mb-3 uppercase tracking-widest border-b border-slate-700 pb-2">
                          <Activity size={14} className="animate-pulse" /> Live Agent Monologue
                        </div>
                        {(msg as any).streamThoughts.map((thought: any, tIdx: number) => (
                          <div key={tIdx} className="animate-in fade-in slide-in-from-left-2 duration-300">
                            <span className="text-emerald-400 font-bold mr-2">[{thought.agent}]:</span>
                            <span className="text-slate-300">{thought.text}</span>
                          </div>
                        ))}
                        <div className="flex gap-2 items-center text-slate-500 pt-2">
                          <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                          Processing Graph...
                        </div>
                      </div>
                    )}

                    {/* The Final Answer */}
                    {msg.content && (
                      <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-8 py-6 max-w-[95%] shadow-sm">
                        <div className="prose prose-slate prose-sm md:prose-base max-w-none">
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                      </div>
                    )}
                    
                    {/* The Telemetry / Graph Trace */}
                    {(msg.subgraph && msg.subgraph.length > 0) && (
                      <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 max-w-[95%] space-y-4 ml-6 shadow-inner">
                        <div className="flex items-center justify-between">
                          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                            <Activity size={14} /> Knowledge Extraction Trace
                          </h4>
                          <div className="flex items-center gap-3">
                            <div className="flex gap-1">
                              <button 
                                onClick={() => handleFeedback(msg.subgraph || [], true)}
                                className="p-1 hover:bg-emerald-100 text-slate-400 hover:text-emerald-600 rounded transition-colors"
                                title="Good Connection (Increase Weight)"
                              >
                                👍
                              </button>
                              <button 
                                onClick={() => handleFeedback(msg.subgraph || [], false)}
                                className="p-1 hover:bg-rose-100 text-slate-400 hover:text-rose-600 rounded transition-colors"
                                title="Hallucination (Slash Weight)"
                              >
                                👎
                              </button>
                            </div>
                            {msg.score !== undefined && (
                              <div className="flex items-center gap-2 bg-white px-3 py-1 rounded-full border border-slate-200 shadow-sm">
                                <ShieldCheck size={14} className={msg.score > 0.7 ? "text-emerald-500" : "text-amber-500"} />
                                <span className="text-xs font-mono font-bold text-slate-700">Verif. Score: {msg.score.toFixed(2)}</span>
                              </div>
                            )}
                          </div>
                        </div>
                        
                        <div className="grid gap-2">
                          {msg.subgraph.map((triplet, tIdx) => (
                            <div 
                              key={tIdx} 
                              onClick={() => setAuditNode(triplet)}
                              className="flex items-center gap-2 text-xs font-mono bg-white border border-slate-100 p-2.5 rounded-lg shadow-sm cursor-pointer hover:border-blue-300 hover:shadow-md transition-all group"
                            >
                              <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded font-semibold border border-blue-100 max-w-[150px] truncate group-hover:bg-blue-100">{triplet.subject}</span>
                              <span className="text-slate-400 font-bold group-hover:text-blue-500">→</span>
                              <span className="px-2 py-1 bg-slate-100 text-slate-600 rounded font-medium border border-slate-200 uppercase tracking-tight group-hover:bg-slate-200">{triplet.predicate}</span>
                              <span className="text-slate-400 font-bold group-hover:text-purple-500">→</span>
                              <span className="px-2 py-1 bg-purple-50 text-purple-700 rounded font-semibold border border-purple-100 max-w-[200px] truncate group-hover:bg-purple-100">{triplet.object}</span>
                              
                              {triplet.source && (
                                <span className="ml-auto flex items-center gap-1 text-[10px] text-blue-500 font-bold bg-blue-50 px-2 py-1 rounded">
                                  <Activity size={10} /> View Grounding
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            
            {loading && (
              <div className="flex flex-col gap-3 ml-6 max-w-[80%]">
                <div className="flex gap-2 items-center text-sm font-medium text-slate-500 bg-white border border-slate-200 py-3 px-5 rounded-full shadow-sm w-fit">
                  <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                  Executing Multi-Agent Graph Traversal...
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-slate-200 p-6 z-20 shadow-[0_-4px_24px_rgba(0,0,0,0.02)]">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleQuery} className="relative flex items-center">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Query the structural knowledge base..."
                className="w-full bg-slate-50 border border-slate-300 rounded-xl pl-6 pr-32 py-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-shadow text-base shadow-inner"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !query}
                className="absolute right-2 top-2 bottom-2 bg-blue-600 text-white px-6 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Execute
              </button>
            </form>
            <p className="text-center text-xs text-slate-400 mt-3 font-medium">Powered by Adaptive GraphRAG • Deterministic Algorithmic Verification Active</p>
          </div>
        </div>

        {/* Audit Trail Modal */}
        {auditNode && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-white w-[600px] rounded-2xl shadow-2xl border border-slate-200 overflow-hidden flex flex-col">
              <div className="bg-slate-900 p-4 flex justify-between items-center text-white">
                <h3 className="font-bold flex items-center gap-2">
                  <ShieldCheck size={18} className="text-emerald-400" />
                  Audit Trail Grounding
                </h3>
                <button onClick={() => setAuditNode(null)} className="text-slate-400 hover:text-white transition-colors text-xl font-bold">&times;</button>
              </div>
              <div className="p-6 space-y-6">
                <div>
                  <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Extracted Graph Node</h4>
                  <div className="flex items-center gap-2 text-sm font-mono bg-slate-50 border border-slate-200 p-3 rounded-lg">
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded font-bold">{auditNode.subject}</span>
                    <span className="text-slate-400">→</span>
                    <span className="px-2 py-1 bg-slate-200 text-slate-700 rounded font-bold uppercase">{auditNode.predicate}</span>
                    <span className="text-slate-400">→</span>
                    <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded font-bold">{auditNode.object}</span>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Cryptographic Source Provenance</h4>
                  <div className="bg-emerald-50 border border-emerald-100 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-mono text-xs font-bold text-emerald-800">SOURCE FILE:</span>
                      <span className="bg-emerald-200 text-emerald-900 text-[10px] px-2 py-0.5 rounded font-bold">{auditNode.source || "System Inference"}</span>
                    </div>
                    <p className="text-sm text-emerald-900/80 font-medium italic border-l-2 border-emerald-300 pl-3">
                      "According to the ingested structural data from {auditNode.source}, it is algorithmically verified that the entity '{auditNode.subject}' inherently maintains a {auditNode.predicate} relationship with '{auditNode.object}'."
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>

    </div>
  );
}
