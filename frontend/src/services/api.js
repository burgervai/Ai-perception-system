import axios from 'axios'

const BASE = '/api'

export const uploadVideo = (file, onProgress) => {
  const fd = new FormData()
  fd.append('file', file)
  return axios.post(`${BASE}/upload`, fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress && onProgress(Math.round(e.loaded * 100 / e.total))
  })
}

export const getTasks = () => axios.get(`${BASE}/tasks`)
export const getTask  = id => axios.get(`${BASE}/tasks/${id}`)
export const getTelemetry = id => axios.get(`${BASE}/telemetry/${id}`)
