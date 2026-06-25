import React, { useState, useRef } from 'react'

export default function UploadPanel({ onTask, apiBase }) {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [filename, setFilename] = useState(null)
  const inputRef = useRef()

  const handleFile = async (file) => {
    if (!file) return
    setFilename(file.name); setError(null); setUploading(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const r = await fetch(`${apiBase}/api/upload`, { method:'POST', body:fd })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const { task_id } = await r.json()
      onTask(task_id)
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div style={S.card}>
      <div style={S.title}>📂 Upload Video</div>
      <div
        style={S.dropzone}
        onDrop={onDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current.click()}
      >
        {uploading ? '⏳ Uploading…' : filename ? `✅ ${filename}` : 'Drop .mp4 here or click'}
      </div>
      <input ref={inputRef} type="file" accept="video/*" style={{display:'none'}}
             onChange={e => handleFile(e.target.files[0])} />
      {error && <div style={S.error}>{error}</div>}
    </div>
  )
}

const S = {
  card: { background:'#21262d', borderRadius:8, padding:12, border:'1px solid #30363d' },
  title: { fontSize:13, fontWeight:600, color:'#58a6ff', marginBottom:8 },
  dropzone: { border:'2px dashed #30363d', borderRadius:6, padding:'18px 10px', textAlign:'center',
               fontSize:12, color:'#8b949e', cursor:'pointer', transition:'border-color .2s',
               ':hover':{ borderColor:'#58a6ff' } },
  error: { marginTop:6, fontSize:11, color:'#f85149' },
}
