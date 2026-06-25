import { useState, useEffect, useRef, useCallback } from 'react'
import { TelemetrySocket } from '../services/websocket.js'
import { getTelemetry } from '../services/api.js'

export function useTelemetry(taskId, maxHistory = 300) {
  const [frames, setFrames]   = useState([])
  const [latest, setLatest]   = useState(null)
  const [connected, setConn]  = useState(false)
  const socketRef = useRef(null)
  const bufRef    = useRef([])

  const addFrame = useCallback(frame => {
    if (frame.type === 'ping') return
    bufRef.current = [...bufRef.current.slice(-maxHistory + 1), frame]
    setFrames([...bufRef.current])
    setLatest(frame)
  }, [maxHistory])

  useEffect(() => {
    if (!taskId) return
    bufRef.current = []
    setFrames([])
    setLatest(null)

    // Pre-load existing telemetry then connect live
    getTelemetry(taskId).then(r => {
      if (r.data.frames?.length) {
        bufRef.current = r.data.frames.slice(-maxHistory)
        setFrames([...bufRef.current])
        setLatest(bufRef.current[bufRef.current.length - 1])
      }
    }).catch(() => {})

    socketRef.current = new TelemetrySocket(taskId, addFrame, () => setConn(false))
    setConn(true)
    return () => socketRef.current?.close()
  }, [taskId, addFrame, maxHistory])

  return { frames, latest, connected }
}
