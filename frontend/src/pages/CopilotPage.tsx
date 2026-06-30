import React, { useState, useRef, useEffect } from 'react';
import { BrainCircuit, Send, User, Bot, Globe2, Search } from 'lucide-react';
import { askCopilot, explainCustomer } from '../api/client';
import type { CopilotMessage } from '../types';
import toast from 'react-hot-toast';

const SUGGESTIONS_EN = [
  'How many theft cases were detected?',
  'What is the model accuracy?',
  'Who are the highest risk customers?',
  'Explain how CNN-LSTM detects theft',
  'What is the average confidence score?',
];

const SUGGESTIONS_AR = [
  'كم عدد حالات السرقة المكتشفة؟',
  'ما هي دقة النموذج؟',
  'من هم العملاء الأكثر مخاطرة؟',
  'كيف يكتشف نموذج CNN-LSTM السرقة؟',
  'ما متوسط درجة الثقة؟',
];

const CopilotPage: React.FC = () => {
  const [messages, setMessages] = useState<CopilotMessage[]>([
    {
      id: '0',
      role: 'assistant',
      content: '👋 **ETD-XAI Copilot** — I can explain predictions, risk scores, model decisions, and theft detection patterns.\n\nType your question below or ask about a specific customer ID. I answer in **English** and **Arabic** (العربية).',
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState('');
  const [customerId, setCustomerId] = useState('');
  const [lang, setLang] = useState<'en' | 'ar'>('en');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async (text?: string, cid?: string) => {
    const q = (text || input).trim();
    if (!q && !cid) return;

    const userMsg: CopilotMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: cid ? `Explain customer: ${cid}` : q,
      timestamp: new Date(),
    };
    setMessages(p => [...p, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const payload: any = { question: q || `Explain this customer`, language: lang };
      if (cid) payload.customer_id = cid;
      const res = await askCopilot(payload);

      const botMsg: CopilotMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: res.data.answer,
        timestamp: new Date(),
      };
      setMessages(p => [...p, botMsg]);
    } catch {
      toast.error('Copilot failed to respond');
      setMessages(p => [...p, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      }]);
    } finally {
      setLoading(false);
    }
  };

  const explainById = async () => {
    if (!customerId.trim()) { toast.error('Enter a Customer ID'); return; }
    await send('', customerId.trim());
    setCustomerId('');
  };

  const MessageBubble = ({ msg }: { msg: CopilotMessage }) => (
    <div className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
        ${msg.role === 'user' ? 'bg-blue-600' : 'bg-purple-600'}`}>
        {msg.role === 'user' ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-white" />}
      </div>
      <div className={`max-w-2xl rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap
        ${msg.role === 'user'
          ? 'bg-blue-600 text-white rounded-tr-sm'
          : 'bg-slate-800 text-slate-200 rounded-tl-sm border border-slate-700'}`}
        dir={lang === 'ar' && msg.role === 'assistant' ? 'rtl' : 'ltr'}
      >
        {msg.content.replace(/\*\*(.*?)\*\*/g, '$1')}
        <p className="text-[10px] opacity-40 mt-1">
          {msg.timestamp.toLocaleTimeString()}
        </p>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto p-6 gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-black text-white flex items-center gap-2">
            <BrainCircuit className="w-7 h-7 text-purple-400" /> AI Copilot
          </h1>
          <p className="text-slate-400 text-sm">XAI explanations in English & Arabic</p>
        </div>
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 rounded-xl p-1">
          <Globe2 className="w-4 h-4 text-slate-400 ml-2" />
          <button onClick={() => setLang('en')} className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${lang === 'en' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>EN</button>
          <button onClick={() => setLang('ar')} className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${lang === 'ar' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>عربي</button>
        </div>
      </div>

      {/* Explain by ID */}
      <div className="flex gap-2 bg-slate-900 border border-slate-800 rounded-xl p-3">
        <Search className="w-4 h-4 text-slate-500 mt-2.5 flex-shrink-0" />
        <input
          value={customerId}
          onChange={e => setCustomerId(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && explainById()}
          placeholder="Enter Customer ID to explain its prediction..."
          className="flex-1 bg-transparent text-sm text-white placeholder:text-slate-500 focus:outline-none font-mono"
        />
        <button onClick={explainById} className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-bold transition-colors">
          Explain
        </button>
      </div>

      {/* Suggestions */}
      <div className="flex gap-2 flex-wrap">
        {(lang === 'en' ? SUGGESTIONS_EN : SUGGESTIONS_AR).map(s => (
          <button key={s} onClick={() => send(s)}
            className="px-3 py-1.5 bg-slate-900 border border-slate-700 hover:border-purple-500/50 hover:text-purple-300 text-slate-400 rounded-xl text-xs font-medium transition-colors"
            dir={lang === 'ar' ? 'rtl' : 'ltr'}
          >
            {s}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 min-h-0 py-2">
        {messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)}
        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <div className="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2 bg-slate-900 border border-slate-700 rounded-xl p-2">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder={lang === 'ar' ? 'اكتب سؤالك هنا...' : 'Ask about predictions, risk scores, model decisions...'}
          dir={lang === 'ar' ? 'rtl' : 'ltr'}
          className="flex-1 bg-transparent text-white text-sm placeholder:text-slate-500 focus:outline-none px-2"
        />
        <button
          onClick={() => send()}
          disabled={!input.trim() || loading}
          className="p-2.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-40 text-white rounded-lg transition-colors"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default CopilotPage;
