import React, { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
         ResponsiveContainer, ReferenceLine } from 'recharts'

export default function PIDChart({ frames }) {
  const data = useMemo(() => {
    let speed = 15
    return frames.slice(-120).map((f, i) => {
      const c = f.control || {}
      speed = Math.max(0, Math.min(30, speed + ((c.throttle||0)-(c.brake||0))*0.4))
      return {
        i,
        distance:  c.current_distance_m  ?? null,
        setpoint:  c.setpoint_m          ?? null,
        error:     c.error_m             ?? null,
        throttle:  c.throttle            ?? null,
        brake:     c.brake != null       ? -c.brake : null,
        speedKmh:  parseFloat((speed*3.6).toFixed(1)),
        latency:   f.ai_latency_ms,
        p:         c.p ?? null,
        i_term:    c.i ?? null,
        d_term:    c.d ?? null,
      }
    })
  }, [frames])

  const tip = { contentStyle:{ background:'#21262d', border:'1px solid #30363d', fontSize:10 } }

  function Chart({ height, children, domain }) {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top:4, right:4, bottom:0, left:-18 }}>
          <CartesianGrid strokeDasharray="2 3" stroke="#21262d"/>
          <XAxis dataKey="i" tick={false} height={0}/>
          <YAxis tick={{ fontSize:8, fill:'#8b949e' }} domain={domain}/>
          <Tooltip {...tip} formatter={(v,n) => [v!=null?Number(v).toFixed(2):'n/a', n]}/>
          {children}
        </LineChart>
      </ResponsiveContainer>
    )
  }

  const sub = (label) => <div style={S.sub}>{label}</div>

  return (
    <div style={S.wrap}>
      <div style={S.title}>📈 Live Telemetry Charts</div>
      <div style={S.scroll}>
        {sub('Distance vs Setpoint (m)')}
        <Chart height={110} domain={[0,'auto']}>
          <Legend wrapperStyle={{fontSize:9}}/>
          <ReferenceLine y={data[0]?.setpoint??10} stroke="#f5a62366" strokeDasharray="3 2"/>
          <Line type="monotone" dataKey="setpoint" stroke="#f5a623" dot={false} name="setpoint" strokeWidth={1.5} strokeDasharray="4 2"/>
          <Line type="monotone" dataKey="distance" stroke="#4e9ef5" dot={false} name="distance" strokeWidth={2} connectNulls/>
        </Chart>

        {sub('Throttle / Brake')}
        <Chart height={90} domain={[-1.05,1.05]}>
          <Legend wrapperStyle={{fontSize:9}}/>
          <ReferenceLine y={0} stroke="#30363d"/>
          <Line type="monotone" dataKey="throttle" stroke="#3fb950" dot={false} name="throttle" strokeWidth={2} connectNulls/>
          <Line type="monotone" dataKey="brake"    stroke="#f85149" dot={false} name="brake"    strokeWidth={2} connectNulls/>
        </Chart>

        {sub('Simulated Speed (km/h)')}
        <Chart height={85} domain={[0,130]}>
          <Line type="monotone" dataKey="speedKmh" stroke="#a78bfa" dot={false} name="speed km/h" strokeWidth={2}/>
        </Chart>

        {sub('Error e(t) (m)')}
        <Chart height={85}>
          <ReferenceLine y={0} stroke="#30363d"/>
          <Line type="monotone" dataKey="error" stroke="#bc8cff" dot={false} name="error" strokeWidth={1.5} connectNulls/>
        </Chart>

        {sub('PID Components')}
        <Chart height={85}>
          <Legend wrapperStyle={{fontSize:9}}/>
          <ReferenceLine y={0} stroke="#30363d"/>
          <Line type="monotone" dataKey="p"      stroke="#f85149" dot={false} name="P" strokeWidth={1.5} connectNulls/>
          <Line type="monotone" dataKey="i_term" stroke="#4e9ef5" dot={false} name="I" strokeWidth={1.5} connectNulls/>
          <Line type="monotone" dataKey="d_term" stroke="#3fb950" dot={false} name="D" strokeWidth={1.5} connectNulls/>
        </Chart>

        {sub('AI Latency (ms)')}
        <Chart height={75}>
          <Line type="monotone" dataKey="latency" stroke="#8b949e" dot={false} name="latency ms" strokeWidth={1}/>
        </Chart>

        {data.length === 0 && <div style={S.empty}>Waiting for telemetry…</div>}
      </div>
    </div>
  )
}

const S = {
  wrap:  { display:'flex', flexDirection:'column', flex:1, overflow:'hidden' },
  title: { fontSize:11, fontWeight:600, color:'#4e9ef5', padding:'6px 10px',
            borderBottom:'1px solid #21262d', flexShrink:0 },
  scroll:{ flex:1, overflowY:'auto', padding:'4px 6px' },
  sub:   { fontSize:9, color:'#8b949e', marginTop:8, marginBottom:2, fontWeight:600, letterSpacing:0.5 },
  empty: { textAlign:'center', color:'#8b949e', fontSize:11, padding:20 },
}
