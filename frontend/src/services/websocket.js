export class TelemetrySocket {
  constructor(taskId, onFrame, onClose) {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host  = window.location.hostname
    this.ws = new WebSocket(`${proto}://${host}:8000/ws/${taskId}`)
    this.ws.onmessage = e => { try { onFrame(JSON.parse(e.data)) } catch {} }
    this.ws.onclose   = () => onClose && onClose()
    this.ws.onerror   = () => onClose && onClose()
  }
  close() { this.ws.close() }
}
