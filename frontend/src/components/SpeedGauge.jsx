/**
 * SpeedGauge — circular SVG speedometer showing:
 *  • Current simulated ego speed (km/h)
 *  • Throttle / brake arc
 *  • Speed history spark line
 */
import React, { useRef, useMemo } from 'react'

const R  = 64    // gauge radius
const CX = 82; const CY = 82
const MIN_A = -210; const MAX_A = 30   // sweep angles in degrees

function degToRad(d) { return d * Math.PI / 180 }
function polarToXY(cx, cy, r, angleDeg) {
  const a = degToRad(angleDeg - 90)
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)]
}
function arc(cx, cy, r, startDeg, endDeg, large) {
  const [x1,y1] = polarToXY(cx,cy,r,startDeg)
  const [x2,y2] = polarToXY(cx,cy,r,endDeg)
  return `M${x1},${y1} A${r},${r} 0 ${large?1:0} 1 ${x2},${y2}`
}

const MAX_SPEED_KMH = 120

export default function SpeedGauge({ ctrl, frames }) {
  const speedRef = useRef(15)

  // Simulate ego speed from last N frames
  const speedHistory = useMemo(() => {
    let speed = 15
    return frames.slice(-60).map(f => {
      const c = f.control || {}
      speed = Math.max(0, Math.min(30, speed + ((c.throttle||0) - (c.brake||0)) * 0.4))
      return speed * 3.6
    })
  }, [frames])

  const speedKmh = speedHistory.length ? speedHistory[speedHistory.length-1] : 54
  const fraction = Math.min(speedKmh / MAX_SPEED_KMH, 1)
  const needleA  = MIN_A + fraction * (MAX_A - MIN_A)

  const thr = ctrl.throttle ?? 0
  const brk = ctrl.brake    ?? 0
  const arcCol = thr > 0.05 ? '#3fb950' : brk > 0.05 ? '#f85149' : '#4e9ef5'

  // Speed spark (bottom strip)
  const spkW = 164; const spkH = 18
  const spkPts = speedHistory.map((v, i) => {
    const x = (i / Math.max(speedHistory.length-1,1)) * spkW
    const y = spkH - (v / MAX_SPEED_KMH) * spkH
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  return (
    <div style={S.wrap}>
      <div style={S.title}>🚗 Speed</div>
      <svg width="100%" viewBox="0 0 164 180" style={{ display:'block' }}>
        {/* BG */}
        <rect width={164} height={180} fill="#0d1117" rx="6"/>

        {/* Track arc */}
        <path d={arc(CX,CY,R+8,MIN_A,MAX_A,1)} stroke="#21262d" strokeWidth="12"
              fill="none" strokeLinecap="round"/>

        {/* Colour zones: green → yellow → red */}
        {[
          [MIN_A,       MIN_A+160, '#1a3a1a'],
          [MIN_A+160,   MIN_A+210, '#3a2a0a'],
          [MIN_A+210,   MAX_A,     '#3a0a0a'],
        ].map(([a1,a2,c],i) => (
          <path key={i} d={arc(CX,CY,R+8,a1,a2,a2-a1>180)} stroke={c}
                strokeWidth="12" fill="none" strokeLinecap="round"/>
        ))}

        {/* Value arc */}
        {needleA > MIN_A && (
          <path d={arc(CX,CY,R+8,MIN_A,needleA,needleA-MIN_A>180)}
                stroke={arcCol} strokeWidth="12" fill="none" strokeLinecap="round" opacity="0.9"/>
        )}

        {/* Tick marks */}
        {Array.from({length:13}, (_,i) => {
          const a   = MIN_A + (i/12) * (MAX_A-MIN_A)
          const r1  = R+2; const r2 = i%3===0 ? R-6 : R-2
          const [x1,y1] = polarToXY(CX,CY,r1,a)
          const [x2,y2] = polarToXY(CX,CY,r2,a)
          const spd = Math.round(i/12 * MAX_SPEED_KMH)
          const [lx,ly] = polarToXY(CX,CY,R-14,a)
          return (
            <g key={i}>
              <line x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={i%3===0?'#8b949e':'#30363d'} strokeWidth={i%3===0?1.5:1}/>
              {i%3===0 && <text x={lx} y={ly+3} fontSize="6.5" fill="#8b949e"
                                textAnchor="middle">{spd}</text>}
            </g>
          )
        })}

        {/* Needle */}
        {(() => {
          const [nx,ny] = polarToXY(CX,CY,R-6,needleA)
          const [bx,by] = polarToXY(CX,CY,10,needleA+180)
          return (
            <g>
              <line x1={bx} y1={by} x2={nx} y2={ny}
                    stroke={arcCol} strokeWidth="2.5" strokeLinecap="round"/>
              <circle cx={CX} cy={CY} r="5" fill={arcCol}/>
              <circle cx={CX} cy={CY} r="2.5" fill="#0d1117"/>
            </g>
          )
        })()}

        {/* Speed readout */}
        <text x={CX} y={CY+22} fontSize="22" fontWeight="bold" fill="#e6edf3"
              textAnchor="middle" fontFamily="monospace">
          {speedKmh.toFixed(0)}
        </text>
        <text x={CX} y={CY+34} fontSize="9" fill="#8b949e" textAnchor="middle">km/h</text>

        {/* Throttle / Brake mini bars */}
        <rect x={12} y={148} width={60} height={6} rx="3" fill="#21262d"/>
        <rect x={12} y={148} width={Math.max(0,thr)*60} height={6} rx="3" fill="#3fb950"/>
        <text x={12} y={145} fontSize="6.5" fill="#3fb950">THR</text>

        <rect x={92} y={148} width={60} height={6} rx="3" fill="#21262d"/>
        <rect x={92} y={148} width={Math.max(0,brk)*60} height={6} rx="3" fill="#f85149"/>
        <text x={92} y={145} fontSize="6.5" fill="#f85149">BRK</text>

        {/* Speed history spark */}
        {speedHistory.length > 1 && (
          <g transform="translate(0,158)">
            <rect width={164} height={spkH+4} fill="#0a0e17"/>
            <polyline points={spkPts} fill="none" stroke={arcCol}
                      strokeWidth="1.5" strokeLinejoin="round" opacity="0.8"/>
            <text x={2} y={spkH+1} fontSize="5.5" fill="#30363d">speed history</text>
          </g>
        )}
      </svg>
    </div>
  )
}

const S = {
  wrap:  { flex:1, background:'#0d1117', borderRight:'1px solid #21262d' },
  title: { fontSize:10, fontWeight:600, color:'#8b949e', padding:'5px 8px',
            borderBottom:'1px solid #21262d' },
}
