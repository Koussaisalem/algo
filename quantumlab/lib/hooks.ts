'use client'

import { useState, useEffect, useCallback } from 'react'

interface FetchState<T> {
  data: T | null
  isLoading: boolean
  error: string | null
}

export function useFetch<T>(url: string, options?: RequestInit) {
  const [state, setState] = useState<FetchState<T>>({
    data: null,
    isLoading: true,
    error: null,
  })

  const fetchData = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }))
    
    try {
      const response = await fetch(url, options)
      
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to fetch')
      }
      
      const data = await response.json()
      setState({ data, isLoading: false, error: null })
    } catch (err) {
      setState({ 
        data: null, 
        isLoading: false, 
        error: err instanceof Error ? err.message : 'Unknown error' 
      })
    }
  }, [url, options])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return { ...state, refetch: fetchData }
}

export function usePolling<T>(url: string, interval: number = 5000) {
  const [state, setState] = useState<FetchState<T>>({
    data: null,
    isLoading: true,
    error: null,
  })

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(url)
        
        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.error || 'Failed to fetch')
        }
        
        const data = await response.json()
        setState({ data, isLoading: false, error: null })
      } catch (err) {
        setState(prev => ({ 
          ...prev, 
          isLoading: false, 
          error: err instanceof Error ? err.message : 'Unknown error' 
        }))
      }
    }

    fetchData()
    const id = setInterval(fetchData, interval)
    
    return () => clearInterval(id)
  }, [url, interval])

  return state
}

export async function apiPost<T>(url: string, data: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || 'Request failed')
  }
  
  return response.json()
}

export async function apiDelete(url: string): Promise<void> {
  const response = await fetch(url, { method: 'DELETE' })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || 'Delete failed')
  }
}

export async function uploadFile(file: File): Promise<unknown> {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || 'Upload failed')
  }
  
  return response.json()
}
