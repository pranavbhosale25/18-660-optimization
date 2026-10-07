"""Offline, dependency-free HTML trajectory viewer (not a physics solver)."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np


def polytope_edges(normals,offsets,tol=1e-7):
    """Small-world host mesh extraction; triple-plane intersections."""
    a,b=np.asarray(normals),np.asarray(offsets)
    vertices=[]
    for inds in combinations(range(len(a)),3):
        ai=a[list(inds)]
        if abs(np.linalg.det(ai))<1e-10: continue
        x=np.linalg.solve(ai,b[list(inds)])
        if (a@x>=b-tol).all() and not any(np.linalg.norm(x-y)<tol for y in vertices):
            vertices.append(x)
    vertices=np.asarray(vertices)
    if not len(vertices): return [],[]
    on=np.abs(a@vertices.T-b[:,None])<tol
    edges=[(i,j) for i,j in combinations(range(len(vertices)),2)
           if np.count_nonzero(on[:,i]&on[:,j])>=2]
    return vertices.tolist(),edges


def write_viewer(path,world,positions,radius,dt,title='Convex-world point-mass simulation'):
    vertices,edges=polytope_edges(world.normals,world.offsets)
    payload={'vertices':vertices,'edges':edges,'positions':np.asarray(positions).tolist(),
             'radius':float(radius),'dt':float(dt)}
    template=r'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Point-mass simulation</title><style>
body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#202937}
h1{font-size:24px;margin-bottom:5px}p{max-width:860px;line-height:1.5}
canvas{width:min(100%,960px);height:600px;background:white;border:1px solid #b6c0cd;border-radius:8px;touch-action:none}
button,input{margin:12px 10px 0 0}input{width:55%;vertical-align:middle}button{padding:8px 16px}
</style><h1 id="title"></h1><p>Drag to rotate the view. The dot marks a point mass (no rotation). Its displayed size is only a visual marker; an optional nonzero collision clearance is supported. This viewer displays a saved trajectory and performs no physics.</p>
<canvas id="c" width="1200" height="750"></canvas><div><button id="play">Pause</button><input id="frame" type="range" min="0" value="0"><span id="time"></span></div>
<script>
const data=PAYLOAD,cv=document.getElementById('c'),ctx=cv.getContext('2d'),slider=document.getElementById('frame');
document.getElementById('title').textContent=TITLE;
slider.max=data.positions.length-1;let playing=true,index=0,yaw=.65,pitch=.35,drag=false,last=[0,0],acc=0,prev=0;
const all=data.vertices.length?data.vertices:data.positions,extent=Math.max(1,...all.map(v=>Math.hypot(...v)));
function project(v){let [x,y,z]=v;let xx=Math.cos(yaw)*x-Math.sin(yaw)*y,yy=Math.sin(yaw)*x+Math.cos(yaw)*y;
let zz=Math.cos(pitch)*z-Math.sin(pitch)*yy,depth=Math.sin(pitch)*z+Math.cos(pitch)*yy;
let scale=270/extent/(1+depth/(5*extent));return [cv.width/2+scale*xx,cv.height/2-scale*zz,scale];}
function line(a,b,style,width){ctx.strokeStyle=style;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a.slice(0,2));ctx.lineTo(...b.slice(0,2));ctx.stroke();}
function draw(){ctx.clearRect(0,0,cv.width,cv.height);for(const [i,j] of data.edges)line(project(data.vertices[i]),project(data.vertices[j]),'#8997a8',1.5);
for(let i=Math.max(1,index-100);i<=index;i++)line(project(data.positions[i-1]),project(data.positions[i]),'#3176af',2);
let p=project(data.positions[index]),r=Math.max(4,data.radius*p[2]);let grd=ctx.createRadialGradient(p[0]-r*.3,p[1]-r*.3,r*.1,p[0],p[1],r);grd.addColorStop(0,'#aecfea');grd.addColorStop(1,'#245b86');ctx.fillStyle=grd;ctx.beginPath();ctx.arc(p[0],p[1],Math.max(2,r),0,2*Math.PI);ctx.fill();
slider.value=index;document.getElementById('time').textContent=`t = ${(index*data.dt).toFixed(3)} s / frame ${index}`;}
cv.onpointerdown=e=>{drag=true;last=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId)};
cv.onpointerup=()=>drag=false;cv.onpointermove=e=>{if(drag){yaw+=(e.clientX-last[0])*.007;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-last[1])*.007));last=[e.clientX,e.clientY];draw();}};
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';};
slider.oninput=()=>{index=+slider.value;draw();};
function tick(t){if(playing){acc+=(t-prev)/1000;while(acc>=data.dt){index=(index+1)%data.positions.length;acc-=data.dt;}draw();}prev=t;requestAnimationFrame(tick);}draw();requestAnimationFrame(t=>{prev=t;requestAnimationFrame(tick);});
</script></html>'''
    # JSON is generated only from numeric data; escape title against HTML script termination.
    text=template.replace('PAYLOAD',json.dumps(payload,separators=(',',':')))
    text=text.replace('TITLE',json.dumps(title).replace('<','\\u003c'))
    Path(path).write_text(text)
