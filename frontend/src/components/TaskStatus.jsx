import React from 'react'

const STATUS_COLOR = { queued:'#8b949e', running:'#58a6ff', done:'#3fb950', error:'#f85149' }

export default function TaskStatus({ status }) {
  const c = STATUS_COLOR[status.status] || '#8b949e'
  const pct = status.n_frames > 0
    ? Math.round((status.current_frame / status.n_frames) * 100)
    : null

  return (
    <div style={S.card}>
      <div style={S.title}>🔄 Task Status</div>
      <div style={{ ...S.badge, background: c+'22', color: c }}>
        {status.status.toUpperCase()}
      </div>
      <div style={S.row}><span style={S.label}>File</span><span style={S.val}>{status.filename}</span></div>
      <div style={S.row}><span style={S.label}>Frames</span><span style={S.val}>{status.n_frames}</span></div>
      {pct !== null && (
        <div style={S.row}>
          <span style={S.label}>Progress</span>
          <div style={S.barWrap}><div style={{...S.bar, width:`${pct}%`, background:c}} /></div>
        </div>
      )}
      {status.error && <div style={S.err}>{status.error}</div>}
    </div>
  )
}

const S = {
  card: { background:'#21262d', borderRadius:8, padding:12, border:'1px solid #30363d' },
  title: { fontSize:13, fontWeight:600, color:'#58a6ff', marginBottom:8 },
  badge: { display:'inline-block', padding:'2px 8px', borderRadius:4, fontSize:11, fontWeight:700, marginBottom:8 },
  row: { display:'flex', alignItems:'center', gap:8, marginBottom:4 },
  label: { fontSize:11, color:'#8b949e', minWidth:55 },
  val: { fontSize:11, color:'#e6edf3' },
  barWrap: { flex:1, height:6, background:'#30363d', borderRadius:3, overflow:'hidden' },
  bar: { height:'100%', borderRadius:3, transition:'width .3s' },
  err: { marginTop:6, fontSize:11, color:'#f85149' },
}
