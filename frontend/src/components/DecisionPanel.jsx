/**
 * DecisionPanel — shows current ego car decision:
 *  • Large ACCELERATE / BRAKE / CRUISE state
 *  • PID component bars (P, I, D)
 *  • Distance error ring
 *  • Lead vehicle info
 */
import React from 'react'

export default function DecisionPanel({ ctrl }) {
  const thr   = ctrl.throttle           ?? 0
  const brk   = ctrl.brake              ?? 0
  const err   = ctrl.error_m            ?? null
  const dist  = ctrl.current_distance_m ?? null
  const setpt = ctrl.setpoint_m         ?? 10
  const p     = ctrl.p                  ?? 0
  const i     = ctrl.i                  ?? 0
  const d     = ctrl.d                  ?? 0
  const ctl   = ctrl.control            ?? 0

  const isAccel  = thr > 0.04
  const isBrake  = brk > 0.04
  const isCruise = !isAccel && !isBrake

  const decLabel = isAccel ? 'ACCELERATE' : isBrake ? 'BRAKE' : 'CRUISE'
  const decColor = isAccel ? '#3fb950'    : isBrake ? '#f85149' : '#8b949e'
  const decIcon  = isAccel ? '▲'         : isBrake ? '▼'      : '⏸'
  const decMag   = isAccel ? thr         : isBrake ? brk      : 0

  // Distance ring: 0=too close, 0.5=perfect, 1=too far
  const distFrac = dist != null ? Math.min(Math.max((dist - 0) / (setpt * 2), 0), 1) : 0.5
  const ringColor = dist != null
    ? dist < setpt * 0.8 ? '#f85149'
    : dist < setpt * 1.2 ? '#3fb950' : '#f5a623'
    : '#30363d'

  function bar(val, max, col) {
    const w = Math.min(Math.abs(val) / Math.max(Math.abs(max), 0.01), 1) * 100
    return (
      <div style={{ height:8, background:'#21262d', borderRadius:4, overflow:'hidden' }}>
        <div style={{ height:'100%', width:`${w}%`, background:col, borderRadius:4,
                      transition:'width 0.2s' }}/>
      </div>
    )
  }

  return (
    <div style={S.wrap}>
      <div style={S.title}>🎯 Decision</div>

      {/* Main decision indicator */}
      <div style={{ ...S.dec, borderColor: decColor + '55', background: decColor + '11' }}>
        <div style={{ fontSize:22, color: decColor }}>{decIcon}</div>
        <div style={{ fontSize:12, fontWeight:700, color: decColor, letterSpacing:1 }}>{decLabel}</div>
        <div style={{ height:6, width:'80%', background:'#21262d', borderRadius:3, overflow:'hidden', marginTop:4 }}>
          <div style={{ height:'100%', width:`${decMag*100}%`, background:decColor,
                        borderRadius:3, transition:'width 0.2s' }}/>
        </div>
        <div style={{ fontSize:9, color: decColor + 'aa', marginTop:2 }}>{(decMag*100).toFixed(0)}%</div>
      </div>

      {/* Distance status */}
      <div style={S.section}>
        <div style={S.sectionTitle}>DISTANCE</div>
        <div style={{ display:'flex', alignItems:'baseline', gap:6 }}>
          <span style={{ fontSize:20, fontWeight:700, color: ringColor, fontFamily:'monospace' }}>
            {dist != null ? dist.toFixed(1) : '--'}
          </span>
          <span style={{ fontSize:10, color:'#8b949e' }}>m</span>
          <span style={{ fontSize:10, color:'#8b949e', marginLeft:'auto' }}>target: {setpt}m</span>
        </div>
        {/* Distance bar */}
        <div style={{ position:'relative', height:10, background:'#21262d', borderRadius:5, overflow:'hidden', marginTop:4 }}>
          {/* Setpoint marker */}
          <div style={{ position:'absolute', left:'50%', top:0, width:2, height:'100%', background:'#ffffff33' }}/>
          {/* Distance fill */}
          <div style={{ height:'100%', width:`${distFrac*100}%`, background:ringColor,
                        borderRadius:5, transition:'width 0.3s' }}/>
        </div>
        {err != null && (
          <div style={{ fontSize:9, color:'#8b949e', marginTop:2 }}>
            error: <span style={{ color: Math.abs(err) > 3 ? '#f5a623' : '#3fb950',
                                  fontFamily:'monospace' }}>{err > 0 ? '+' : ''}{err.toFixed(2)}m</span>
          </div>
        )}
      </div>

      {/* PID component bars */}
      <div style={S.section}>
        <div style={S.sectionTitle}>PID COMPONENTS</div>
        {[
          { label:'P (proportional)', val:p, max:0.5, col:'#f85149' },
          { label:'I (integral)',     val:i, max:0.2, col:'#4e9ef5' },
          { label:'D (derivative)',   val:d, max:0.15, col:'#3fb950' },
        ].map(({ label, val, max, col }) => (
          <div key={label} style={{ marginBottom:5 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:2 }}>
              <span style={{ fontSize:9, color:'#8b949e' }}>{label}</span>
              <span style={{ fontSize:9, color:col, fontFamily:'monospace' }}>
                {val > 0 ? '+' : ''}{val.toFixed(4)}
              </span>
            </div>
            {bar(val, max, col)}
          </div>
        ))}
        <div style={{ display:'flex', justifyContent:'space-between', marginTop:4, paddingTop:4, borderTop:'1px solid #21262d' }}>
          <span style={{ fontSize:9, color:'#8b949e' }}>Output u(t)</span>
          <span style={{ fontSize:10, fontFamily:'monospace',
                         color: ctl > 0 ? '#3fb950' : ctl < 0 ? '#f85149' : '#8b949e' }}>
            {ctl >= 0 ? '+' : ''}{ctl.toFixed(4)}
          </span>
        </div>
        {/* Output bar (bidirectional) */}
        <div style={{ position:'relative', height:8, background:'#21262d', borderRadius:4, overflow:'hidden', marginTop:3 }}>
          <div style={{ position:'absolute', left:'50%', top:0, width:1, height:'100%', background:'#ffffff22' }}/>
          <div style={{
            position:'absolute',
            left:  ctl >= 0 ? '50%' : `${(0.5 + ctl/2)*100}%`,
            width: `${Math.abs(ctl)*50}%`,
            height:'100%',
            background: ctl >= 0 ? '#3fb950' : '#f85149',
            transition:'all 0.2s',
          }}/>
        </div>
      </div>
    </div>
  )
}

const S = {
  wrap:        { flex:1, background:'#0d1117', display:'flex', flexDirection:'column', overflow:'hidden' },
  title:       { fontSize:10, fontWeight:600, color:'#8b949e', padding:'5px 8px', borderBottom:'1px solid #21262d', flexShrink:0 },
  dec:         { display:'flex', flexDirection:'column', alignItems:'center', padding:'10px 6px',
                 margin:8, borderRadius:8, border:'1px solid', gap:2 },
  section:     { padding:'6px 10px', borderBottom:'1px solid #21262d' },
  sectionTitle:{ fontSize:8, fontWeight:600, color:'#8b949e', letterSpacing:1, marginBottom:5 },
}
