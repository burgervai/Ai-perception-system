import React, { useState, useEffect, useRef, useCallback } from 'react'
import UploadPanel   from './components/UploadPanel'
import TaskStatus    from './components/TaskStatus'
import Scene3D       from './components/Scene3D'
import BirdsEyeView  from './components/BirdsEyeView'
import SpeedGauge    from './components/SpeedGauge'
import DecisionPanel from './components/DecisionPanel'
import PIDChart      from './components/PIDChart'
import TrackTable    from './components/TrackTable'

const API_BASE = `http://${window.location.hostname}:8000`
const WS_BASE  = `ws://${window.location.hostname}:8000`

export default function App() {
  const [taskId,     setTaskId]     = useState(null)
  const [taskStatus, setTaskStatus] = useState(null)
  const [frames,     setFrames]     = useState([])
  const [latest,     setLatest]     = useState(null)
  const wsRef   = useRef(null)
  const pollRef = useRef(null)

  const addFrame = useCallback(frame => {
    if (frame.type === 'ping') return
    setLatest(frame)
    setFrames(prev => { const n = [...prev, frame]; return n.length > 300 ? n.slice(-300) : n })
  }, [])

  useEffect(() => {
    if (!taskId) return
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/tasks/${taskId}`)
        const d = await r.json()
        setTaskStatus(d)
        if (d.status === 'done' || d.status === 'error') clearInterval(pollRef.current)
      } catch {}
    }, 1000)
    return () => clearInterval(pollRef.current)
  }, [taskId])

  useEffect(() => {
    if (!taskId) return
    if (wsRef.current) wsRef.current.close()
    const ws = new WebSocket(`${WS_BASE}/ws/${taskId}`)
    wsRef.current = ws
    ws.onmessage = e => { try { addFrame(JSON.parse(e.data)) } catch {} }
    return () => ws.close()
  }, [taskId, addFrame])

  const ctrl  = latest?.control  || {}
  const tracks = latest?.tracks  || []

  return (
    <div style={S.root}>
      <header style={S.header}>
        <span style={S.logo}>⚡ AI Perception System</span>
        <span style={S.sep}/>
        <span style={S.tag}>Pure-Python Nodes</span>
        <span style={S.tag}>EKF Tracker</span>
        <span style={S.tag}>PID Cruise Control</span>
        <span style={S.tag}>MLflow</span>
        {latest && <span style={S.live}>● LIVE  frame {latest.frame_id}</span>}
      </header>

      <div style={S.body}>
        {/* ── Left sidebar ─────────────────────────────── */}
        <div style={S.sidebar}>
          <UploadPanel onTask={setTaskId} apiBase={API_BASE} />
          {taskStatus && <TaskStatus status={taskStatus} />}
          <TrackTable tracks={tracks} />
        </div>

        {/* ── Centre: 3D world + bird's eye ────────────── */}
        <div style={S.centre}>
          <div style={S.scene3dWrap}>
            <Scene3D latestFrame={latest} frames={frames} />
          </div>
          <div style={S.birdsWrap}>
            <BirdsEyeView latestFrame={latest} frames={frames} />
          </div>
        </div>

        {/* ── Right panel: gauges + charts ─────────────── */}
        <div style={S.right}>
          <div style={S.gaugesRow}>
            <SpeedGauge ctrl={ctrl} frames={frames} />
            <DecisionPanel ctrl={ctrl} />
          </div>
          <PIDChart frames={frames} />
        </div>
      </div>
    </div>
  )
}

const S = {
  root:       { display:'flex', flexDirection:'column', height:'100vh', background:'#0a0e17', color:'#e6edf3', overflow:'hidden' },
  header:     { display:'flex', alignItems:'center', gap:10, padding:'8px 16px', background:'#0d1117', borderBottom:'1px solid #21262d', flexShrink:0 },
  logo:       { fontSize:16, fontWeight:700, color:'#58a6ff', marginRight:4 },
  sep:        { flex:1 },
  tag:        { fontSize:10, padding:'2px 7px', borderRadius:4, background:'#21262d', color:'#8b949e', border:'1px solid #30363d' },
  live:       { fontSize:11, color:'#3fb950', fontWeight:600, marginLeft:8 },
  body:       { display:'flex', flex:1, overflow:'hidden', gap:0 },
  sidebar:    { width:252, flexShrink:0, display:'flex', flexDirection:'column', gap:8, padding:8, background:'#0d1117', borderRight:'1px solid #21262d', overflowY:'auto' },
  centre:     { flex:1, display:'flex', flexDirection:'column', minWidth:0 },
  scene3dWrap:{ flex:'0 0 58%', position:'relative', background:'#0a0e17' },
  birdsWrap:  { flex:'0 0 42%', position:'relative', borderTop:'1px solid #21262d' },
  right:      { width:300, flexShrink:0, display:'flex', flexDirection:'column', background:'#0d1117', borderLeft:'1px solid #21262d', overflow:'hidden' },
  gaugesRow:  { display:'flex', gap:0, flexShrink:0, borderBottom:'1px solid #21262d' },
}
