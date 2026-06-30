import React, { useEffect, useState, useCallback } from 'react';
import {
  useReactTable, getCoreRowModel, getSortedRowModel,
  flexRender, createColumnHelper, type SortingState,
} from '@tanstack/react-table';
import {
  Search, Download, Filter, ChevronUp, ChevronDown,
  ChevronLeft, ChevronRight, AlertTriangle, CheckCircle2
} from 'lucide-react';
import { getCustomers, exportCSV, exportJSON } from '../api/client';
import type { Customer } from '../types';
import toast from 'react-hot-toast';

const col = createColumnHelper<Customer>();

const StatusBadge = ({ status }: { status: string }) =>
  status === 'Theft' ? (
    <span className="flex items-center gap-1 px-2 py-0.5 bg-red-500/15 text-red-400 rounded-full text-xs font-bold border border-red-500/20">
      <AlertTriangle className="w-3 h-3" />Theft
    </span>
  ) : (
    <span className="flex items-center gap-1 px-2 py-0.5 bg-emerald-500/15 text-emerald-400 rounded-full text-xs font-bold border border-emerald-500/20">
      <CheckCircle2 className="w-3 h-3" />Normal
    </span>
  );

const RiskBar = ({ score }: { score: number }) => (
  <div className="flex items-center gap-2">
    <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full ${score > 75 ? 'bg-red-500' : score > 50 ? 'bg-orange-500' : score > 25 ? 'bg-yellow-500' : 'bg-emerald-500'}`}
        style={{ width: `${score}%` }}
      />
    </div>
    <span className="text-white text-xs w-8 text-right">{score.toFixed(0)}</span>
  </div>
);

const CustomersPage: React.FC = () => {
  const [data, setData] = useState<Customer[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [totalPages, setTotalPages] = useState(1);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [sortBy, setSortBy] = useState('risk_score');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [loading, setLoading] = useState(false);
  const [sorting, setSorting] = useState<SortingState>([]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getCustomers({ page, page_size: pageSize, search, status_filter: statusFilter, sort_by: sortBy, sort_dir: sortDir });
      setData(res.data.data);
      setTotal(res.data.total);
      setTotalPages(res.data.total_pages);
    } catch {
      toast.error('Failed to load customers');
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, search, statusFilter, sortBy, sortDir]);

  useEffect(() => { load(); }, [load]);

  // Sync TanStack sorting to API sort params
  useEffect(() => {
    if (sorting.length > 0) {
      setSortBy(sorting[0].id);
      setSortDir(sorting[0].desc ? 'desc' : 'asc');
    }
  }, [sorting]);

  const columns = [
    col.accessor('id', {
      header: 'Customer ID',
      cell: i => <span className="font-mono text-xs text-blue-300">{i.getValue()}</span>,
    }),
    col.accessor('status', {
      header: 'Status',
      cell: i => <StatusBadge status={i.getValue()} />,
    }),
    col.accessor('probability', {
      header: 'Probability',
      cell: i => <span className="font-mono text-sm text-white">{(i.getValue() * 100).toFixed(2)}%</span>,
    }),
    col.accessor('confidence', {
      header: 'Confidence',
      cell: i => <span className="font-mono text-sm text-yellow-300">{(i.getValue() * 100).toFixed(2)}%</span>,
    }),
    col.accessor('risk_score', {
      header: 'Risk Score',
      cell: i => <RiskBar score={i.getValue()} />,
    }),
    col.accessor('flag', {
      header: 'Ground Truth',
      cell: i => {
        const v = i.getValue();
        if (v === null || v === undefined) return <span className="text-slate-600 text-xs">N/A</span>;
        return v === 1
          ? <span className="text-red-400 text-xs font-bold">Theft</span>
          : <span className="text-emerald-400 text-xs font-bold">Normal</span>;
      },
    }),
  ];

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    manualSorting: true,
    manualPagination: true,
    pageCount: totalPages,
  });

  const handleExport = async (type: 'csv' | 'json') => {
    try {
      const res = type === 'csv' ? await exportCSV() : await exportJSON();
      const url = URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement('a');
      a.href = url;
      a.download = `etd_predictions.${type}`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported as ${type.toUpperCase()}`);
    } catch {
      toast.error('Export failed');
    }
  };

  return (
    <div className="p-6 space-y-5 max-w-screen-2xl mx-auto">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-black text-white">Customer Predictions</h1>
          <p className="text-slate-400 text-sm">{total.toLocaleString()} customers processed by CNN-LSTM</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => handleExport('csv')} className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-sm transition-colors">
            <Download className="w-3.5 h-3.5" /> CSV
          </button>
          <button onClick={() => handleExport('json')} className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-sm transition-colors">
            <Download className="w-3.5 h-3.5" /> JSON
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative flex-1 min-w-56">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1); }}
            placeholder="Search customer ID..."
            className="w-full pl-9 pr-4 py-2.5 bg-slate-900 border border-slate-700 text-white rounded-xl text-sm focus:outline-none focus:border-blue-500 placeholder:text-slate-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-500" />
          {['', 'Theft', 'Normal'].map(f => (
            <button key={f} onClick={() => { setStatusFilter(f); setPage(1); }}
              className={`px-3 py-2 rounded-xl text-xs font-bold transition-colors
                ${statusFilter === f
                  ? f === 'Theft' ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                    : f === 'Normal' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}>
              {f || 'All'}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              {table.getHeaderGroups().map(hg => (
                <tr key={hg.id} className="border-b border-slate-800">
                  {hg.headers.map(header => (
                    <th key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      className="text-left px-4 py-3 text-slate-400 font-semibold text-xs uppercase tracking-wider cursor-pointer hover:text-white select-none"
                    >
                      <div className="flex items-center gap-1">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getIsSorted() === 'asc' ? <ChevronUp className="w-3 h-3" /> :
                          header.column.getIsSorted() === 'desc' ? <ChevronDown className="w-3 h-3" /> : null}
                      </div>
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={6} className="text-center py-8 text-slate-500">Loading...</td></tr>
              ) : table.getRowModel().rows.map(row => (
                <tr key={row.id} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                  {row.getVisibleCells().map(cell => (
                    <td key={cell.id} className="px-4 py-3">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-slate-800">
          <p className="text-slate-400 text-xs">
            Showing {((page - 1) * pageSize + 1).toLocaleString()}–{Math.min(page * pageSize, total).toLocaleString()} of {total.toLocaleString()}
          </p>
          <div className="flex items-center gap-2">
            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-white transition-colors">
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-slate-400 text-xs px-2">Page {page} of {totalPages}</span>
            <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-white transition-colors">
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CustomersPage;
