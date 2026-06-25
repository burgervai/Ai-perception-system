/**
 * BirdsEyeView — Top-down SVG showing:
 *  • Ego car at bottom centre
 *  • Lead + all tracks at their (x_offset, distance) positions
 *  • Trajectory trail of the ego car's simulated path
 *  • Setpoint distance ring
 *  • Velocity vectors on each track
 *  • Lateral offset indicator
 */
import React, { useRef, useMemo } from 'react'

const CLS_COLOR = {
  car:'#4e9ef5', truck:'#f5a623', pedestrian:'#50c878',
  bicyclist:'#ffcb00', bus:'#e05252', van:'#a78bfa',
  rider:'#34d399', others:'#94a3b8', trafficcone:'#fb923c',
}

const VIEW_W  = 200   // virtual metres width shown
const VIEW_D  = 80    // virtual metres depth shown (forward)
const PAD     = 12

export default function BirdsEyeView({ latestFrame, frames }) {
  const svgRef = useRef(null)

  // Build trajectory from frames (simulated ego X, Z)
  const trajectory = useMemo(() => {
    let speed = 15, posX = 0, posZ = 0, yaw = 0
    const pts = []
    frames.slice(-150).forEach(f => {
      const ctrl = f.control || {}
      const thr  = ctrl.throttle ?? 0
      const brk  = ctrl.brake ?? 0
      speed = Math.max(0, Math.min(30, speed + (thr - brk) * 0.4))
      posX += speed * 0.1 * Math.sin(yaw)
      posZ += speed * 0.1 * Math.cos(yaw)
      pts.push({ x: posX, z: posZ, speed, thr, brk })
    })
    return pts
  }, [frames])

  const ctrl   = latestFrame?.control || {}
  const tracks = latestFrame?.tracks  || []
  const setpt  = ctrl.setpoint_m ?? 10
  const curDist= ctrl.current_distance_m
  const thr    = ctrl.throttle ?? 0
  const brk    = ctrl.brake    ?? 0

  // Current ego position from last trajectory point
  const egoPt = trajectory.length ? trajectory[trajectory.length-1] : { x:0, z:0, speed:15 }
  const egoSpeedKmh = (egoPt.speed * 3.6).toFixed(0)

  // SVG coordinate helpers
  // World: x=lateral (right+), z=forward (+)
  // SVG: (0,0) top-left; ego at centre-bottom
  const svgW = 280, svgH = 340
  const cx   = svgW / 2
  const cy   = svgH - 40

  function worldToSvg(wx, wz) {
    const sx = cx + (wx / (VIEW_W/2)) * (svgW/2 - PAD)
    const sy = cy - (wz / VIEW_D) * (svgH - 60)
    return [sx, sy]
  }

  // Relative trajectory points (recent 60)
  const relTraj = trajectory.slice(-60).map((pt, i) => {
    const rx = pt.x - egoPt.x
    const rz = pt.z - egoPt.z
    return worldToSvg(rx, rz)
  })

  const trailPath = relTraj.length > 1
    ? 'M' + relTraj.map(([x,y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join('L')
    : ''

  // Setpoint arc
  const [spx1, spy1] = worldToSvg(-4, setpt)
  const [spx2, spy2] = worldToSvg( 4, setpt)

  return (
    <div style={S.wrap}>
      <div style={S.title}>🛣️ Bird's Eye · Trajectory · Object Map</div>
      <div style={S.inner}>
        <svg ref={svgRef} width="100%" viewBox={`0 0 ${svgW} ${svgH}`}
             style={{ display:'block', height:'100%' }}>
          {/* Background */}
          <rect width={svgW} height={svgH} fill="#0a0e17" rx="4"/>

          {/* Road surface */}
          <rect x={cx-30} y={PAD} width={60} height={svgH-PAD-10} fill="#1a1f2e" rx="3"/>

          {/* Lane markings */}
          {Array.from({length:12}, (_,i) => {
            const yy = PAD + 15 + i * 24
            return <rect key={i} x={cx-0.8} y={yy} width={1.6} height={12} fill="#ffffff44" rx="0.5"/>
          })}

          {/* Road edges */}
          <line x1={cx-30} y1={PAD} x2={cx-30} y2={svgH-10} stroke="#ffcc0066" strokeWidth="1.5"/>
          <line x1={cx+30} y1={PAD} x2={cx+30} y2={svgH-10} stroke="#ffcc0066" strokeWidth="1.5"/>

          {/* Distance ruler (right side) */}
          {[10,20,30,40,50,60].map(d => {
            const [,sy] = worldToSvg(0, d)
            if (sy < PAD || sy > svgH-10) return null
            return <g key={d}>
              <line x1={svgW-28} y1={sy} x2={svgW-22} y2={sy} stroke="#30363d" strokeWidth="0.8"/>
              <text x={svgW-20} y={sy+3} fontSize="7" fill="#8b949e">{d}m</text>
            </g>
          })}

          {/* Setpoint ring */}
          {(() => {
            const [,sy] = worldToSvg(0, setpt)
            if (sy >= PAD && sy <= svgH-10) return (
              <g>
                <line x1={cx-35} y1={sy} x2={cx+35} y2={sy}
                      stroke="#f85149" strokeWidth="1" strokeDasharray="4,3" opacity="0.7"/>
                <text x={cx+37} y={sy+3} fontSize="7" fill="#f85149">{setpt}m target</text>
              </g>
            )
          })()}

          {/* Trajectory trail */}
          {trailPath && (
            <path d={trailPath} stroke="#4e9ef5" strokeWidth="2" fill="none"
                  strokeLinecap="round" strokeLinejoin="round"
                  opacity="0.55" strokeDasharray="3,2"/>
          )}
          {/* Trail dots (every 5th point coloured by speed) */}
          {relTraj.filter((_,i) => i % 5 === 0).map(([x,y], i) => {
            const t   = trajectory[trajectory.length - 60 + i*5]
            const spd = t ? t.speed/30 : 0.5
            const col = `hsl(${240 - spd*140}, 80%, 60%)`
            return <circle key={i} cx={x.toFixed(1)} cy={y.toFixed(1)} r="2.5"
                           fill={col} opacity="0.8"/>
          })}

          {/* Tracked objects */}
          {tracks.map(t => {
            const [sx, sy] = worldToSvg(t.x_offset * 2, t.z)
            if (sy < PAD || sy > svgH-5) return null
            const col = CLS_COLOR[t.class_name] || '#8b949e'
            const dim = { car:[7,12], truck:[8,18], bus:[10,22], pedestrian:[3,5],
                          bicyclist:[3,6], van:[7,14] }[t.class_name] || [5,9]
            // Velocity arrow
            const vlen = Math.min(Math.abs(t.vz) * 4, 20)
            const arrowY = sy - (t.vz < 0 ? vlen : -vlen)
            return (
              <g key={t.track_id}>
                {/* Object box */}
                <rect x={sx-dim[0]/2} y={sy-dim[1]/2} width={dim[0]} height={dim[1]}
                      fill={col+'33'} stroke={col} strokeWidth="1.5" rx="1.5"/>
                {/* Velocity arrow */}
                {Math.abs(t.vz) > 0.3 && (
                  <line x1={sx} y1={sy} x2={sx} y2={arrowY}
                        stroke={t.vz < 0 ? '#f85149' : '#3fb950'}
                        strokeWidth="2" markerEnd="url(#arr)"/>
                )}
                {/* Label */}
                <text x={sx+dim[0]/2+2} y={sy+3} fontSize="7" fill={col}>
                  {t.class_name[0].toUpperCase()}{t.track_id} {t.z.toFixed(0)}m
                </text>
                {/* Track ID dot */}
                <circle cx={sx} cy={sy} r="1.5" fill={col}/>
              </g>
            )
          })}

          {/* Arrow marker def */}
          <defs>
            <marker id="arr" markerWidth="5" markerHeight="5" refX="3" refY="2.5" orient="auto">
              <polygon points="0,0 5,2.5 0,5" fill="#ffffff66"/>
            </marker>
          </defs>

          {/* Ego car */}
          <g>
            <rect x={cx-9} y={cy-16} width={18} height={26}
                  fill="#2244aa" stroke="#4e9ef5" strokeWidth="2" rx="3"/>
            <rect x={cx-6}  y={cy-13} width={12} height={14}
                  fill="#1a3388" rx="2"/>
            {/* Headlights */}
            <circle cx={cx-7} cy={cy-16} r="2" fill="#ffffaa" opacity="0.9"/>
            <circle cx={cx+7} cy={cy-16} r="2" fill="#ffffaa" opacity="0.9"/>
            {/* Taillights */}
            <rect x={cx-8} y={cy+8} width={4} height={2.5} fill="#ff4444" rx="1"/>
            <rect x={cx+4} y={cy+8} width={4} height={2.5} fill="#ff4444" rx="1"/>
          </g>

          {/* Lateral offset indicator (bottom bar) */}
          {ctrl.current_distance_m != null && (() => {
            const lead  = tracks.find(t => t.track_id === ctrl.lead_track_id)
            const xoff  = lead ? lead.x_offset : 0
            const bx    = Math.max(cx-55, Math.min(cx+55, cx + xoff * 18))
            return (
              <g>
                <text x={cx} y={svgH-4} fontSize="7" fill="#8b949e" textAnchor="middle">
                  lateral offset: {xoff.toFixed(2)}m
                </text>
                <line x1={cx-55} y1={svgH-14} x2={cx+55} y2={svgH-14}
                      stroke="#30363d" strokeWidth="1"/>
                <circle cx={bx} cy={svgH-14} r="3.5" fill="#f5a623"/>
                <line x1={cx} y1={svgH-17} x2={cx} y2={svgH-11}
                      stroke="#30363d" strokeWidth="1" strokeDasharray="2,1"/>
              </g>
            )
          })()}

          {/* Speed & decision overlay (top-right) */}
          <rect x={svgW-68} y={PAD} width={60} height={36} fill="#0d1117cc" rx="4"/>
          <text x={svgW-38} y={PAD+13} fontSize="9" fill="#4e9ef5" textAnchor="middle" fontWeight="bold">
            {egoSpeedKmh} km/h
          </text>
          <text x={svgW-38} y={PAD+26} fontSize="7.5" fill={thr>0.05?'#3fb950':brk>0.05?'#f85149':'#8b949e'}
                textAnchor="middle">
            {thr>0.05?'▲ ACCEL':brk>0.05?'▼ BRAKE':'─ CRUISE'}
          </text>
        </svg>

        {/* Legend */}
        <div style={S.legend}>
          {Object.entries(CLS_COLOR).map(([cls, col]) => (
            <span key={cls} style={{ ...S.chip, borderColor: col, color: col }}>{cls}</span>
          ))}
          <span style={{ ...S.chip, borderColor:'#4e9ef5', color:'#4e9ef5' }}>── trail</span>
          <span style={{ ...S.chip, borderColor:'#f85149', color:'#f85149' }}>── target</span>
        </div>
      </div>
    </div>
  )
}

const S = {
  wrap:   { width:'100%', height:'100%', display:'flex', flexDirection:'column', overflow:'hidden' },
  title:  { fontSize:11, fontWeight:600, color:'#4e9ef5', padding:'5px 10px', borderBottom:'1px solid #21262d', flexShrink:0 },
  inner:  { flex:1, display:'flex', flexDirection:'column', overflow:'hidden', padding:4 },
  legend: { display:'flex', flexWrap:'wrap', gap:3, padding:'4px 6px', flexShrink:0 },
  chip:   { fontSize:9, padding:'1px 5px', borderRadius:3, border:'1px solid', background:'#0a0e17' },
}
