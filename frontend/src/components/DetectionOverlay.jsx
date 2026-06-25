import React from 'react'

const CLS_COLOR = { car:'#58a6ff', pedestrian:'#3fb950', cyclist:'#d29922' }

export default function DetectionOverlay({ detections, frameId }) {
  if (!detections.length) return null
  // Overlay is pure SVG, positioned over the 3D canvas
  return (
    <div style={S.wrap}>
      <div style={S.header}>
        🔍 Frame {frameId} — {detections.length} detection{detections.length!==1?'s':''}
      </div>
      <div style={S.grid}>
        {detections.map((d, i) => (
          <div key={i} style={{ ...S.det, borderColor: CLS_COLOR[d.class_name]||'#8b949e' }}>
            <span style={{ color: CLS_COLOR[d.class_name]||'#e6edf3', fontWeight:600 }}>
              {d.class_name}
            </span>
            <span style={S.val}>{d.distance_m.toFixed(1)} m</span>
            <span style={S.score}>{(d.score*100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

const S = {
  wrap: { padding:'6px 10px', background:'#161b22', borderTop:'1px solid #30363d' },
  header: { fontSize:11, color:'#8b949e', marginBottom:4 },
  grid: { display:'flex', flexWrap:'wrap', gap:4 },
  det: { display:'flex', gap:6, alignItems:'center', padding:'2px 8px', borderRadius:4,
          border:'1px solid', fontSize:11, background:'#21262d' },
  val: { color:'#e6edf3' },
  score: { color:'#8b949e' },
}
