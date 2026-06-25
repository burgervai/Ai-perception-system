import React, { useEffect, useState } from 'react'
import { getTasks } from '../services/api.js'

const STATUS_COLOR = { queued:'#8b949e', running:'#d29922', done:'#3fb950', error:'#f85149' }

export default function TaskSidebar({ activeTask, onSelect }) {
  const [tasks, setTasks] = useState([])

  useEffect(() => {
    const refresh = () => getTasks().then(r => setTasks(r.data)).catch(() => {})
    refresh()
    const id = setInterval(refresh, 2000)
    return () => clearInterval(id)
  }, [])

  return (
    <div style={styles.sidebar}>
      <div style={styles.header}>Recent Tasks</div>
      {tasks.length === 0 && <div style={styles.empty}>No tasks yet</div>}
      {tasks.map(t => (
        <div key={t.task_id} style={{...styles.item, ...(t.task_id===activeTask ? styles.activeItem : {})}}
             onClick={() => onSelect(t.task_id)}>
          <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:2}}>
            <span style={{...styles.dot, background: STATUS_COLOR[t.status]??'#8b949e'}} />
            <span style={styles.fname}>{t.filename}</span>
          </div>
          <div style={styles.meta}>
            <span style={{color:STATUS_COLOR[t.status]}}>{t.status}</span>
            {t.status==='running' && <span> — frame {t.current_frame}</span>}
            {t.status==='done' && <span> — {t.n_frames} frames</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

const styles = {
  sidebar: { width:220, background:'#161b22', borderRight:'1px solid #30363d', overflowY:'auto', display:'flex', flexDirection:'column' },
  header:  { padding:'0.85rem 1rem', fontWeight:700, color:'#e6edf3', fontSize:'0.85rem', borderBottom:'1px solid #21262d', letterSpacing:'0.04em' },
  empty:   { color:'#30363d', fontSize:'0.8rem', padding:'0.8rem 1rem' },
  item:    { padding:'0.65rem 1rem', cursor:'pointer', borderBottom:'1px solid #0d1117', transition:'background .15s' },
  activeItem: { background:'#1f2937' },
  fname:   { color:'#e6edf3', fontSize:'0.82rem', fontWeight:500, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', maxWidth:160 },
  meta:    { color:'#6e7681', fontSize:'0.75rem', paddingLeft:14 },
  dot:     { width:7, height:7, borderRadius:'50%', flexShrink:0 },
}
