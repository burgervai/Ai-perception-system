"""Generate a synthetic dashcam video for end-to-end testing."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import cv2, numpy as np


def _project(X,Y,Z,f,H,W):
    if Z<=0.1: Z=0.1
    return int(W/2+(X/Z)*f), int(H/2-((Y-1.6)/Z)*f)


def _draw_object(img, cls, x, z, f, H, W):
    dims={"car":(1.8,1.5),"pedestrian":(0.5,1.7),"cyclist":(0.7,1.7)}
    w_obj,h_obj=dims.get(cls,(1.5,1.5))
    pixels=[_project(xx,yy,zz,f,H,W) for xx in [x-w_obj/2,x+w_obj/2] for zz in [z-0.5,z+0.5] for yy in [0,h_obj]]
    us,vs=[p[0] for p in pixels],[p[1] for p in pixels]
    u1,u2=max(0,min(us)),min(W-1,max(us)); v1,v2=max(0,min(vs)),min(H-1,max(vs))
    if u2-u1<4 or v2-v1<4: return
    colors={"car":(50,70,220),"pedestrian":(200,140,50),"cyclist":(60,200,200)}
    cv2.rectangle(img,(u1,v1),(u2,v2),colors.get(cls,(120,120,120)),thickness=-1)
    cv2.rectangle(img,(u1,v1),(u2,v2),(0,0,0),thickness=1)


def render_frame(t, H=480, W=640, f=720.):
    img=np.zeros((H,W,3),dtype=np.uint8)
    for v in range(H):
        r=v/H; img[v,:]=(int(140*(1-r)+70*r),int(180*(1-r)+70*r),int(180*(1-r)+80*r))
    cv2.line(img,(0,H//2+10),(W,H//2+10),(110,110,110),1)
    lane_y=H//2+10
    for i in range(8):
        y=lane_y+int((i**1.5)*6)
        if y>=H: break
        cv2.rectangle(img,(W//2-3,y),(W//2+3,y+5),(255,255,255),-1)
    lead_z=15.+9.*math.sin(2*math.pi*t/12.)
    _draw_object(img,"car",0.,lead_z,f,H,W)
    side_x=-2.5*math.sin(2*math.pi*t/8.)
    _draw_object(img,"car",side_x,18.,f,H,W)
    _draw_object(img,"pedestrian",1.*math.sin(2*math.pi*t/20.),30.,f,H,W)
    cv2.putText(img,f"t={t:5.1f}s",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,f"lead~{lead_z:5.1f}m",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1,cv2.LINE_AA)
    return img


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="data/sample/drive.mp4")
    p.add_argument("--duration-s",type=float,default=24.)
    p.add_argument("--fps",type=int,default=20)
    p.add_argument("--width",type=int,default=640)
    p.add_argument("--height",type=int,default=480)
    args=p.parse_args()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(args.out,cv2.VideoWriter_fourcc(*"mp4v"),args.fps,(args.width,args.height))
    if not writer.isOpened(): raise SystemExit(f"cannot open writer for {args.out}")
    n=int(args.duration_s*args.fps); print(f"Rendering {n} frames -> {args.out}")
    for i in range(n): writer.write(render_frame(i/args.fps,args.height,args.width))
    writer.release(); print("done.")

if __name__=="__main__": main()
