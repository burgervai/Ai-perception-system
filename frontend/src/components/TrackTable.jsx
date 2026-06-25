import React from 'react'

const CLS_COLOR = { car:'#58a6ff', pedestrian:'#3fb950', cyclist:'#d29922' }

export default function TrackTable({ tracks }) {
  if (!tracks.length)
    return <div style={S.empty}>No active tracks</div>

  return (
    <div style={S.card}>
      <div style={S.title}>🎯 Tracked Objects ({tracks.length})</div>
      <table style={S.table}>
        <thead><tr>
          {['ID','Class','Dist m','Vel m/s'].map(h => <th key={h} style={S.th}>{h}</th>)}
        </tr></thead>
        <tbody>
          {tracks.map(t => (
            <tr key={t.track_id}>
              <td style={S.td}>{t.track_id}</td>
              <td style={{...S.td, color: CLS_COLOR[t.class_name]||'#e6edf3'}}>{t.class_name}</td>
              <td style={S.td}>{t.distance_m.toFixed(1)}</td>
              <td style={{...S.td, color: t.vz < -0.5 ? '#f85149' : t.vz > 0.5 ? '#3fb950' : '#e6edf3'}}>
                {t.vz.toFixed(1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const S = {
  card: { background:'#21262d', borderRadius:8, padding:12, border:'1px solid #30363d' },
  title: { fontSize:13, fontWeight:600, color:'#58a6ff', marginBottom:8 },
  table: { width:'100%', borderCollapse:'collapse', fontSize:11 },
  th: { textAlign:'left', color:'#8b949e', padding:'3px 6px', borderBottom:'1px solid #30363d' },
  td: { padding:'3px 6px', color:'#e6edf3' },
  empty: { fontSize:11, color:'#8b949e', padding:8, textAlign:'center' },
}
