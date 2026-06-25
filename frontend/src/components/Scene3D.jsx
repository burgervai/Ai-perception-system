/**
 * Scene3D — full Three.js 3D world showing:
 *  • Ego car driving on a road (camera chase view)
 *  • Road scrolling at ego-simulated speed
 *  • Lead car and other tracks as 3D boxes at correct distances
 *  • Trajectory trail (last 80 positions as ribbon on road)
 *  • Speed-responsive road blur / lane marking flicker
 *  • HUD overlay (distance ring, setpoint arc)
 */
import React, { useRef, useEffect } from 'react'
import * as THREE from 'three'

const CLS_HEX = { car:0x4e9ef5, truck:0xf5a623, pedestrian:0x50c878,
                  bicyclist:0xffcb00, bus:0xe05252, van:0xa78bfa,
                  rider:0x34d399, others:0x94a3b8, trafficcone:0xfb923c }
const CLS_DIMS = { car:[1.8,1.5,4.2], truck:[2.4,3.5,7], pedestrian:[0.5,1.7,0.5],
                   bicyclist:[0.6,1.7,1.2], bus:[2.5,3.2,11], van:[2,2,5],
                   rider:[0.6,1.7,1], others:[1,1.5,1], trafficcone:[0.3,0.7,0.3] }

// Physics simulation for ego car
const MAX_SPEED  = 30    // m/s
const MAX_ACCEL  = 4     // m/s²
const ROAD_W     = 12
const ROAD_LEN   = 300

export default function Scene3D({ latestFrame, frames }) {
  const mountRef = useRef(null)
  const stRef    = useRef({})    // three.js state
  const phyRef   = useRef({      // physics state
    speed:     15,
    posX:      0,
    posZ:      0,
    yaw:       0,
    roadOff:   0,
    trailPts:  [],
  })

  /* ── Setup Three.js (once) ──────────────────────────────────────────── */
  useEffect(() => {
    const el  = mountRef.current
    const W   = el.clientWidth
    const H   = el.clientHeight

    const renderer = new THREE.WebGLRenderer({ antialias:true })
    renderer.setSize(W, H)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.setClearColor(0x0a0e17)
    el.appendChild(renderer.domElement)

    const scene  = new THREE.Scene()
    scene.fog    = new THREE.FogExp2(0x0a0e17, 0.008)

    // Camera (chase view behind ego)
    const camera = new THREE.PerspectiveCamera(55, W/H, 0.1, 600)
    camera.position.set(0, 5, -12)
    camera.lookAt(0, 1, 30)

    // Lights
    scene.add(new THREE.AmbientLight(0x334466, 0.8))
    const sun = new THREE.DirectionalLight(0x6699cc, 1.4)
    sun.position.set(20, 60, 30)
    sun.castShadow = true
    scene.add(sun)
    const rim = new THREE.PointLight(0x4466ff, 0.6, 80)
    rim.position.set(-10, 8, 10)
    scene.add(rim)

    // ── Road ──────────────────────────────────────────────────────────
    const roadMat = new THREE.MeshLambertMaterial({ color:0x1a1f2e })
    const road    = new THREE.Mesh(new THREE.PlaneGeometry(ROAD_W, ROAD_LEN), roadMat)
    road.rotation.x = -Math.PI/2
    road.position.set(0, 0, ROAD_LEN/2 - 20)
    road.receiveShadow = true
    scene.add(road)

    // Road shoulders
    ;[-ROAD_W/2-1, ROAD_W/2+1].forEach(x => {
      const sh = new THREE.Mesh(
        new THREE.PlaneGeometry(2, ROAD_LEN),
        new THREE.MeshLambertMaterial({ color:0x2a2f3e }))
      sh.rotation.x = -Math.PI/2
      sh.position.set(x, 0, ROAD_LEN/2 - 20)
      scene.add(sh)
    })

    // Lane markings — group so we can scroll them
    const laneGrp = new THREE.Group()
    scene.add(laneGrp)
    const dashMat = new THREE.MeshBasicMaterial({ color:0xffffff })
    const edgeMat = new THREE.MeshBasicMaterial({ color:0xffcc00 })

    for (let i = 0; i < 50; i++) {
      // Centre dashes
      const dash = new THREE.Mesh(new THREE.PlaneGeometry(0.15, 3), dashMat)
      dash.rotation.x = -Math.PI/2
      dash.position.set(0, 0.01, i * 8)
      laneGrp.add(dash)
      // Side dashes
      ;[-ROAD_W/2+0.15, ROAD_W/2-0.15].forEach(x => {
        const side = new THREE.Mesh(new THREE.PlaneGeometry(0.25, 3), edgeMat)
        side.rotation.x = -Math.PI/2
        side.position.set(x, 0.01, i * 8)
        laneGrp.add(side)
      })
    }

    // ── Ego car body ──────────────────────────────────────────────────
    const egoGroup = new THREE.Group()
    scene.add(egoGroup)
    egoGroup.position.set(0, 0, 0)

    const bodyMat = new THREE.MeshLambertMaterial({ color:0x2244aa })
    const egoBody = new THREE.Mesh(new THREE.BoxGeometry(1.8, 1.3, 4.2), bodyMat)
    egoBody.position.set(0, 0.75, 0)
    egoGroup.add(egoBody)

    const roofMat = new THREE.MeshLambertMaterial({ color:0x1a3388 })
    const egoRoof = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.7, 2.2), roofMat)
    egoRoof.position.set(0, 1.75, -0.3)
    egoGroup.add(egoRoof)

    // Headlights
    ;[[-0.7, 0.55, 2.1], [0.7, 0.55, 2.1]].forEach(([x,y,z]) => {
      const hl = new THREE.PointLight(0xffffaa, 0.8, 25)
      hl.position.set(x, y, z)
      egoGroup.add(hl)
      const lens = new THREE.Mesh(
        new THREE.SphereGeometry(0.12, 8, 8),
        new THREE.MeshBasicMaterial({ color:0xffffee }))
      lens.position.set(x, y, z)
      egoGroup.add(lens)
    })

    // Wheel arches
    ;[[-0.95,0.3,1.2],[0.95,0.3,1.2],[-0.95,0.3,-1.2],[0.95,0.3,-1.2]].forEach(([x,y,z]) => {
      const wheel = new THREE.Mesh(
        new THREE.CylinderGeometry(0.3, 0.3, 0.22, 16),
        new THREE.MeshLambertMaterial({ color:0x111111 }))
      wheel.rotation.z = Math.PI/2
      wheel.position.set(x, y, z)
      egoGroup.add(wheel)
    })

    // Edge lines on ego
    const egoEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1.8,1.3,4.2)),
      new THREE.LineBasicMaterial({ color:0x4466ff, linewidth:1 }))
    egoEdges.position.set(0, 0.75, 0)
    egoGroup.add(egoEdges)

    // ── Trajectory trail (BufferGeometry ribbon) ──────────────────────
    const TRAIL_LEN = 80
    const trailPositions = new Float32Array(TRAIL_LEN * 3)
    const trailGeom = new THREE.BufferGeometry()
    trailGeom.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3))
    const trailMat  = new THREE.LineBasicMaterial({
      color:0x4466ff, transparent:true, opacity:0.6 })
    const trail = new THREE.Line(trailGeom, trailMat)
    scene.add(trail)

    // Small dot markers along trail
    const dotGrp = new THREE.Group(); scene.add(dotGrp)
    const dotMat = new THREE.MeshBasicMaterial({ color:0x4466ff })
    const dots   = Array.from({length: TRAIL_LEN}, (_, i) => {
      const d = new THREE.Mesh(new THREE.SphereGeometry(0.06, 4, 4), dotMat)
      d.visible = false
      dotGrp.add(d)
      return d
    })

    // ── Object pool for tracked vehicles ─────────────────────────────
    const pool = Array.from({length: 30}, () => {
      const grp   = new THREE.Group()
      const box   = new THREE.Mesh(new THREE.BoxGeometry(1,1,1),
                      new THREE.MeshLambertMaterial({ color:0x58a6ff, transparent:true, opacity:0.85 }))
      const edges = new THREE.LineSegments(
                      new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),
                      new THREE.LineBasicMaterial({ color:0xffffff }))
      grp.add(box); grp.add(edges)
      grp.visible = false
      scene.add(grp)
      return { grp, box, edges }
    })

    // ── Setpoint ring (distance target indicator) ─────────────────────
    const ringGeo  = new THREE.RingGeometry(0.3, 0.5, 32)
    ringGeo.rotateX(-Math.PI/2)
    const ringMesh = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color:0xff4444, side:THREE.DoubleSide }))
    ringMesh.position.set(0, 0.02, 10)
    scene.add(ringMesh)

    // ── Setpoint distance label (HTML overlay) ────────────────────────
    const hud = document.createElement('div')
    hud.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none'
    el.appendChild(hud)

    // ── Grid floor ────────────────────────────────────────────────────
    const gridHelper = new THREE.GridHelper(200, 30, 0x1a2040, 0x1a2040)
    gridHelper.position.set(0, 0, 80)
    scene.add(gridHelper)

    stRef.current = { renderer, scene, camera, pool, trail, trailPositions, dots,
                      laneGrp, ringMesh, hud, egoGroup }

    const onResize = () => {
      const W2 = el.clientWidth; const H2 = el.clientHeight
      camera.aspect = W2/H2; camera.updateProjectionMatrix()
      renderer.setSize(W2, H2)
    }
    window.addEventListener('resize', onResize)

    let raf
    const animate = () => { raf = requestAnimationFrame(animate); renderer.render(scene, camera) }
    animate()

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      if (mountRef.current) {
        while (mountRef.current.firstChild) mountRef.current.removeChild(mountRef.current.firstChild)
      }
    }
  }, [])

  /* ── Update physics + scene each new frame ───────────────────────── */
  useEffect(() => {
    const st  = stRef.current
    const phy = phyRef.current
    if (!st.pool || !latestFrame) return

    const ctrl   = latestFrame.control  || {}
    const tracks = latestFrame.tracks   || []
    const thr    = ctrl.throttle        ?? 0
    const brk    = ctrl.brake           ?? 0
    const setpt  = ctrl.setpoint_m      ?? 10
    const dt     = 0.1

    // ── Physics: update ego speed & position ─────────────────────────
    phy.speed = Math.max(2, Math.min(MAX_SPEED, phy.speed + (thr - brk) * MAX_ACCEL * dt))  // min 2 m/s keeps road scrolling
    const dx  = phy.speed * dt * Math.sin(phy.yaw)
    const dz  = phy.speed * dt * Math.cos(phy.yaw)
    phy.posX += dx; phy.posZ += dz
    phy.roadOff = (phy.roadOff + phy.speed * dt) % 8

    // Mild yaw from any nearby lead object's x_offset
    const lead = tracks.find(t => Math.abs(t.x_offset) < 3)
    if (lead) phy.yaw += (-lead.x_offset * 0.002 - phy.yaw) * 0.05

    // ── Trail ─────────────────────────────────────────────────────────
    phy.trailPts.unshift({ x: phy.posX, z: phy.posZ })
    if (phy.trailPts.length > 80) phy.trailPts.pop()

    // Map trail into world-space (relative to ego, behind it)
    const tp = st.trailPositions
    phy.trailPts.forEach(({ x, z }, i) => {
      const wx = -(x - phy.posX)          // negate so trail is behind
      const wz = -(z - phy.posZ)
      tp[i*3]   = wx; tp[i*3+1] = 0.05; tp[i*3+2] = wz
    })
    // Fill remainder
    for (let i = phy.trailPts.length; i < 80; i++) { tp[i*3]=0; tp[i*3+1]=0.05; tp[i*3+2]=0 }
    st.trail.geometry.attributes.position.needsUpdate = true
    st.trail.geometry.setDrawRange(0, phy.trailPts.length)

    st.dots.forEach((d, i) => {
      if (i < phy.trailPts.length) {
        d.position.set(tp[i*3], 0.06, tp[i*3+2])
        d.visible = (i % 5 === 0)
      } else { d.visible = false }
    })

    // ── Lane scrolling ────────────────────────────────────────────────
    st.laneGrp.position.z = phy.roadOff

    // ── Setpoint ring position ────────────────────────────────────────
    st.ringMesh.position.z = setpt

    // ── Tracked objects ───────────────────────────────────────────────
    st.pool.forEach(p => { p.grp.visible = false })

    tracks.slice(0, st.pool.length).forEach((t, i) => {
      const { grp, box, edges } = st.pool[i]
      const [w, h, d] = CLS_DIMS[t.class_name] || [1.5, 1.5, 2]
      const hex       = CLS_HEX[t.class_name]  || 0x8b949e
      const z         = Math.max(2, Math.min(t.z, 120))
      const x         = t.x_offset * 2   // amplify lateral for visibility

      // Rebuild geometry if size changed
      if (box.userData.w !== w || box.userData.h !== h || box.userData.d !== d) {
        box.geometry.dispose()
        box.geometry    = new THREE.BoxGeometry(w, h, d)
        edges.geometry.dispose()
        edges.geometry  = new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d))
        Object.assign(box.userData, { w, h, d })
      }
      box.material.color.setHex(hex)
      box.material.opacity = 0.85
      // Lerp position for smooth movement instead of instant jumps
      grp.position.x += (x       - grp.position.x) * 0.15
      grp.position.y += (h / 2   - grp.position.y) * 0.15
      grp.position.z += (z       - grp.position.z) * 0.15

      // Pulse effect for approaching cars
      if (t.vz < -1) {
        const pulse = 0.8 + 0.2 * Math.sin(Date.now() * 0.008)
        box.material.opacity = pulse
      }
      grp.visible = true
    })

    // ── HUD labels ────────────────────────────────────────────────────
    if (st.hud) {
      st.hud.innerHTML = ''
      tracks.forEach(t => {
        const pos = new THREE.Vector3(t.x_offset*2, (CLS_DIMS[t.class_name]||[0,1.5])[1]/2, Math.min(t.z,120))
        pos.project(st.camera)
        const rect = st.renderer.domElement.getBoundingClientRect()
        const px   = (pos.x*0.5+0.5)*rect.width
        const py   = (-pos.y*0.5+0.5)*rect.height
        if (pos.z >= 1 || px < 0 || px > rect.width) return
        const col  = '#' + (CLS_HEX[t.class_name]||0x8b949e).toString(16).padStart(6,'0')
        const lbl  = document.createElement('div')
        lbl.style.cssText = `position:absolute;left:${px}px;top:${py-18}px;transform:translateX(-50%);
          font:10px monospace;color:#fff;background:${col}cc;padding:1px 5px;border-radius:3px;white-space:nowrap`
        lbl.textContent = `${t.class_name} ${t.distance_m.toFixed(1)}m  ${t.vz<0?'▼':'▲'}${Math.abs(t.vz).toFixed(1)}`
        st.hud.appendChild(lbl)
      })

      // Speed readout
      const spd = document.createElement('div')
      spd.style.cssText = `position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
        font:bold 12px monospace;color:#4e9ef5;background:#0a0e17cc;padding:3px 12px;border-radius:4px;
        border:1px solid #21262d`
      spd.textContent = `${(phy.speed*3.6).toFixed(0)} km/h  |  setpoint ${setpt}m  |  trail ${phy.trailPts.length}pts`
      st.hud.appendChild(spd)

      // Decision tag
      const dec = thr > 0.05 ? '🔼 ACCELERATE' : brk > 0.05 ? '🔽 BRAKE' : '⏸ CRUISE'
      const dcol= thr > 0.05 ? '#3fb950' : brk > 0.05 ? '#f85149' : '#8b949e'
      const dtag = document.createElement('div')
      dtag.style.cssText = `position:absolute;top:8px;left:50%;transform:translateX(-50%);
        font:bold 11px monospace;color:${dcol};background:#0a0e17cc;padding:2px 12px;border-radius:4px;
        border:1px solid ${dcol}55`
      dtag.textContent = dec
      st.hud.appendChild(dtag)
    }
  }, [latestFrame])

  return (
    <div style={{ position:'relative', width:'100%', height:'100%' }}>
      <div style={{ position:'absolute', top:6, left:8, zIndex:5, fontSize:10, color:'#4e9ef5' }}>
        3D World View · chase cam
      </div>
      <div ref={mountRef} style={{ width:'100%', height:'100%' }} />
    </div>
  )
}
