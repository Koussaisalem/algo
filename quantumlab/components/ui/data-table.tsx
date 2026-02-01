'use client';

import { useState, useMemo, ReactNode } from 'react';
import { ChevronUp, ChevronDown, Search, ChevronLeft, ChevronRight } from 'lucide-react';

interface Column<T> {
  key: keyof T | string;
  label: string;
  sortable?: boolean;
  render?: (item: T) => ReactNode;
  className?: string;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  searchable?: boolean;
  searchKeys?: (keyof T)[];
  pageSize?: number;
  onRowClick?: (item: T) => void;
  emptyMessage?: string;
}

export function DataTable<T extends { id: string | number }>({
  data,
  columns,
  searchable = false,
  searchKeys = [],
  pageSize = 10,
  onRowClick,
  emptyMessage = 'No data available',
}: DataTableProps<T>) {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(0);

  const filteredData = useMemo(() => {
    if (!search || searchKeys.length === 0) return data;
    const query = search.toLowerCase();
    return data.filter((item) =>
      searchKeys.some((key) => {
        const value = item[key];
        return String(value).toLowerCase().includes(query);
      })
    );
  }, [data, search, searchKeys]);

  const sortedData = useMemo(() => {
    if (!sortKey) return filteredData;
    return [...filteredData].sort((a, b) => {
      const aVal = (a as Record<string, unknown>)[sortKey];
      const bVal = (b as Record<string, unknown>)[sortKey];
      if (aVal === bVal) return 0;
      const cmp = aVal! > bVal! ? 1 : -1;
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [filteredData, sortKey, sortDir]);

  const paginatedData = useMemo(() => {
    const start = page * pageSize;
    return sortedData.slice(start, start + pageSize);
  }, [sortedData, page, pageSize]);

  const totalPages = Math.ceil(sortedData.length / pageSize);

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  return (
    <div className="glass rounded-2xl border border-white/10 overflow-hidden">
      {searchable && (
        <div className="p-4 border-b border-white/10">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(0);
              }}
              placeholder="Search..."
              className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/40 focus:outline-none focus:border-blue-500/50 transition-colors"
            />
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/10">
              {columns.map((col) => (
                <th
                  key={String(col.key)}
                  className={`px-4 py-3 text-left text-sm font-medium text-white/60 ${col.className || ''}`}
                >
                  {col.sortable ? (
                    <button
                      onClick={() => handleSort(String(col.key))}
                      className="flex items-center gap-1 hover:text-white transition-colors"
                    >
                      {col.label}
                      {sortKey === col.key ? (
                        sortDir === 'asc' ? (
                          <ChevronUp className="w-4 h-4" />
                        ) : (
                          <ChevronDown className="w-4 h-4" />
                        )
                      ) : (
                        <ChevronDown className="w-4 h-4 opacity-30" />
                      )}
                    </button>
                  ) : (
                    col.label
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedData.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-4 py-12 text-center text-white/40"
                >
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              paginatedData.map((item) => (
                <tr
                  key={item.id}
                  onClick={() => onRowClick?.(item)}
                  className={`border-b border-white/5 transition-colors ${
                    onRowClick ? 'cursor-pointer hover:bg-white/5' : ''
                  }`}
                >
                  {columns.map((col) => (
                    <td
                      key={String(col.key)}
                      className={`px-4 py-4 text-sm text-white ${col.className || ''}`}
                    >
                      {col.render
                        ? col.render(item)
                        : String((item as Record<string, unknown>)[String(col.key)] ?? '')}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-white/10">
          <span className="text-sm text-white/60">
            Showing {page * pageSize + 1}-{Math.min((page + 1) * pageSize, sortedData.length)} of {sortedData.length}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="p-2 rounded-lg hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-4 h-4 text-white/60" />
            </button>
            {[...Array(Math.min(5, totalPages))].map((_, i) => {
              const pageNum = Math.max(0, Math.min(page - 2, totalPages - 5)) + i;
              if (pageNum >= totalPages) return null;
              return (
                <button
                  key={pageNum}
                  onClick={() => setPage(pageNum)}
                  className={`w-8 h-8 rounded-lg text-sm font-medium transition-colors ${
                    page === pageNum
                      ? 'bg-blue-500 text-white'
                      : 'text-white/60 hover:bg-white/5'
                  }`}
                >
                  {pageNum + 1}
                </button>
              );
            })}
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="p-2 rounded-lg hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight className="w-4 h-4 text-white/60" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
