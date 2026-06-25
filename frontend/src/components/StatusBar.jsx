import React from 'react'

export default function StatusBar({ task, connected, latest }) {
  const dot = connected ? '#3fb950' : '#f85149'
  return (
    <div style={styles.bar}>
      <span style={{...styles.dot, background: dot}} />
      <span style={styles.label}>{connected ? 'Live' : 'Disconnected'}</span>
      {task && <span style={styles.info}>Task: <code style={styles.code}>{task.slice(0,8)}…</code></span>}
      {latest && <>
        <span style={styles.sep}>|</span>
        <span style={styles.info}>Frame <b>{latest.frame_id}</b></span>
        <span style={styles.sep}>|</span>
        <span style={styles.info}>AI <b>{latest.ai_latency_ms?.toFixed(1)}ms</b></span>
        {latest.control?.current_distance_m != null && <>
          <span style={styles.sep}>|</span>
          <span style={styles.info}>Lead <b>{latest.control.current_distance_m.toFixed(1)}m</b></span>
          <span style={styles.sep}>|</span>
          <span style={styles.info}>
            Throttle <b style={{color:'#3fb950'}}>{(latest.control.throttle*100).toFixed(0)}%</b>
            {' '} Brake <b style={{color:'#f85149'}}>{(latest.control.brake*100).toFixed(0)}%</b>
          </span>
        </>}
      </>}
    </div>
  )
}

const styles = {
  bar:  { display:'flex', alignItems:'center', gap:'0.6rem', padding:'0.45rem 1rem', background:'#161b22', borderBottom:'1px solid #30363d', fontSize:'0.82rem', color:'#8b949e', flexWrap:'wrap' },
  dot:  { width:8, height:8, borderRadius:'50%', display:'inline-block', flexShrink:0 },
  label:{ fontWeight:600, color:'#e6edf3' },
  info: { color:'#8b949e' },
  sep:  { color:'#30363d' },
  code: { background:'#21262d', padding:'0 4px', borderRadius:4, fontFamily:'monospace', fontSize:'0.78rem', color:'#58a6ff' },
}
