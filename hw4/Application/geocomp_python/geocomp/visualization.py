"""Portable 3-D mesh exports and a self-contained, offline Plotly dashboard."""

from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
from plotly.offline import get_plotlyjs


def export_instance(instance, result, validation, out_dir, n_realizations=6, seed=874):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b = instance.basis
    p = instance.problem
    x = result.x
    command = np.asarray(instance.command_coefficients(x))
    mean0 = np.asarray(instance.mean_coefficients(np.zeros_like(x)))
    mean1 = np.asarray(instance.mean_coefficients(x))
    # Common random numbers show compensation, not different random luck.
    key = jax.random.PRNGKey(seed)
    samples0 = np.asarray(
        instance.sample_coefficients(np.zeros_like(x), key, n_realizations)
    )
    samples1 = np.asarray(instance.sample_coefficients(x, key, n_realizations))
    coeffs = {
        "target": np.zeros(p.n_features),
        "command_optimized": command,
        "mean_uncompensated": mean0,
        "mean_optimized": mean1,
    }
    for k in range(n_realizations):
        coeffs[f"realization_uncompensated_{k+1:02d}"] = samples0[k]
        coeffs[f"realization_optimized_{k+1:02d}"] = samples1[k]
    diagnostics = {}
    for name, c in coeffs.items():
        b.mesh(c).export(out / (name + ".stl"))
        diagnostics[name] = b.diagnostics(c)
    # Preserve float64 coordinates, control basis, process parameters, and draws.
    extra_arrays = b.extra_arrays() if hasattr(b, "extra_arrays") else {}
    np.savez_compressed(
        out / "arrays.npz",
        target_vertices=b.vertices,
        faces=b.faces,
        normals=b.normals,
        weights=b.weights,
        anchors=b.anchors,
        anchor_indices=b.anchor_indices,
        x=x,
        command_coefficients=command,
        mean_uncompensated=mean0,
        mean_optimized=mean1,
        samples_uncompensated=samples0,
        samples_optimized=samples1,
        bias=p.bias,
        response=p.response,
        covariance=p.covariance,
        lower=p.lower,
        upper=p.upper,
        Q=p.Q,
        bounds=p.bounds,
        tolerance_mm=instance.command_scale_mm,
        **extra_arrays,
    )
    (out / "mesh_checks.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    # Viewer serializes vector deltas; amplification never modifies exported STLs.
    def disp(c):
        return np.asarray(b.displacement(jnp.asarray(c))).round(9).tolist()

    payload = {
        "name": instance.name,
        "process": instance.process,
        "shape": instance.shape,
        "description": instance.description,
        "vertices": b.vertices.round(9).tolist(),
        "faces": b.faces.tolist(),
        "anchors": b.anchors.round(9).tolist(),
        "command": disp(command),
        "mean0": disp(mean0),
        "mean1": disp(mean1),
        "samples0": [disp(v) for v in samples0],
        "samples1": [disp(v) for v in samples1],
        "tolerance_mm": instance.command_scale_mm,
        "validation": validation,
        "objective": result.objective,
        "iterations": result.iterations,
        "status": result.status,
        "target_probability": p.alpha,
        "working_probability": result.working_target,
        "optimized_probability": result.probability,
        "gap": result.absolute_gap,
        "history": result.history,
        "units": "mm",
    }
    if hasattr(instance, "export_extras"):
        payload["context"] = instance.export_extras(out, x, samples0, samples1)
    (out / "viewer_data.json").write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    return payload


def write_dashboard(results_dir, destination=None):
    """Write a dashboard and return its path, or return ``None`` if no case data exist."""
    root = Path(results_dir)
    destination = Path(destination) if destination else root / "viewer.html"
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(root.glob("*/viewer_data.json"))
    ]
    if not records:
        return None
    data = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")
    html = HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", data)
    destination.write_text(html, encoding="utf-8")
    return destination


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Geometry Compensation Lab — JAX / Veinott</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#18252c;background:#f5f6f7}
body{margin:0}header{padding:28px 4vw 22px;background:white;border-bottom:1px solid #d8dfe3}
.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;font-weight:700}
h1{font-size:32px;letter-spacing:-1px;margin:8px 0}p{line-height:1.55;margin:8px 0}
main{padding:20px 4vw 40px}.tools{display:flex;flex-wrap:wrap;gap:18px;align-items:end;margin-bottom:18px}
label{display:block;font-size:12px;font-weight:650;margin-bottom:6px}
select{font:inherit;padding:9px;border:1px solid #b9c6ce;border-radius:5px;background:white;min-width:130px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.metric,.panel{background:white;border:1px solid #d8dfe3;border-radius:7px}
.metric{padding:16px}.metric span{font-size:12px;display:block}.metric b{font-size:24px;display:block;margin-top:6px}
.meshes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.panel h2{font-size:16px;margin:16px 18px 2px}.panel p{font-size:12px;margin:0 18px 8px;min-height:18px}
.plot{height:430px}.notes{font-size:13px;background:white;padding:18px;border:1px solid #d8dfe3;border-radius:7px;margin:16px 0}
#convergence{height:310px;background:white}#description{max-width:1050px;font-size:14px}
.status{padding:10px 13px;border-left:3px solid #526570;background:#eef1f3;font-size:13px;margin-bottom:15px}
footer{font-size:12px;line-height:1.6;margin-top:18px}code{font-family:ui-monospace,monospace}
@media(max-width:800px){.meshes{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.plot{height:360px}h1{font-size:25px}}
</style><script>__PLOTLY__</script></head><body>
<header><div class="eyebrow">Computational manufacturing / research prototype</div>
<h1>Geometry Compensation Lab</h1>
<p>Joint Gaussian chance constraints · JAX derivatives · Veinott boundary cuts</p>
<p id="description"></p></header><main>
<div class="tools"><div><label for="case">Process / target geometry</label><select id="case"></select></div>
<div><label for="realization">Produced geometry</label><select id="realization"><option value="mean">Mean prediction</option></select></div>
<div><label for="scale">Display deformation</label><select id="scale"><option value="1">1× — true scale</option><option value="2">2× — visual only</option><option value="4">4× — visual only</option><option value="8">8× — visual only</option><option value="20">20× — visual only</option></select></div>
<div><label for="context">Embedded-print context</label><select id="context"><option value="yes">Bath + nozzle centerlines</option><option value="no">Meshes only</option></select></div>
<div><label for="overlay">Target reference</label><select id="overlay"><option value="yes">Show target overlay</option><option value="no">Hide target overlay</option></select></div></div>
<div class="status" id="status"></div><div class="metrics">
<div class="metric"><span>Required joint yield</span><b id="required"></b></div>
<div class="metric"><span>Validated coefficient-box yield</span><b id="box"></b><span id="box-ci"></span></div>
<div class="metric"><span>Validated whole-mesh yield</span><b id="mesh"></b><span id="mesh-ci"></span></div>
<div class="metric"><span>Uniform displacement tolerance</span><b id="tol"></b></div></div>
<div class="meshes">
<section class="panel"><h2>01 / Target</h2><p>Desired manufactured surface, in millimeters.</p><div class="plot" id="target"></div></section>
<section class="panel"><h2>02 / Optimized command geometry</h2><p id="command-caption">Pre-distorted design surface — not the expected finished part.</p><div class="plot" id="command"></div></section>
<section class="panel"><h2>03 / Produced without compensation</h2><p id="baseline-caption"></p><div class="plot" id="baseline"></div></section>
<section class="panel"><h2>04 / Produced with compensation</h2><p id="optimized-caption"></p><div class="plot" id="optimized"></div></section></div>
<div class="notes"><p id="geometry-note"></p><strong>Read the two yields separately.</strong> The optimized event bounds every deformation coefficient simultaneously.
Nonnegative interpolation, together with the model-specific displacement budgets, makes this event sufficient to bound displacement over the entire corresponding triangle mesh.
The directly simulated mesh event can therefore have a higher yield. Intervals are individual 95% Clopper–Pearson intervals from independent simulations;
they do not cover process-model error. The 0.4-percentage-point optimization buffer is numerical headroom, not a proof of integration accuracy.</div>
<section class="panel"><h2>Optimization trace</h2><p>Upper cost is from a numerically feasible incumbent; lower cost comes from the current LP. With numerical probability cuts, this gap is diagnostic, not a certified global bound.</p><div id="convergence"></div></section>
<footer><strong>Scope:</strong> Synthetic affine fixed-covariance process surrogates, not a DLP exposure solver, CNC toolpath simulator, or support-bath fluid solver. Embedded tubes keep target cross-section frames fixed, model equivalent circular radii, and do not predict breakup, fusion or gelation.
No topology optimization, cutter accessibility, support design, collision checking, or self-intersection guarantee. Camera views are linked.
Deformation amplification affects display only; STL exports retain actual millimeter coordinates (STL itself does not encode units).</footer>
</main><script>
const DATA=__DATA__;
const ids=['target','command','baseline','optimized'];
let camera={eye:{x:1.55,y:-1.8,z:1.2}},lock=false;
const USE_WEBGL=(()=>{try{let c=document.createElement('canvas');return !!(c.getContext('webgl2')||c.getContext('webgl'));}catch(e){return false;}})();
let cpu={yaw:-.7,pitch:.55,zoom:1},cpuScene=null;
function bathSegments(bounds){let corners=Array.from({length:8},(_,i)=>[0,1,2].map(j=>bounds[(i>>j)&1][j]));let edges=[];
 for(let i=0;i<8;i++)for(let j=0;j<3;j++){let k=i^(1<<j);if(i<k)edges.push([corners[i],corners[k]]);}return edges;}
function pathTrace(points,name){return {type:'scatter3d',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),line:{width:4},name,hoverinfo:'name'};}
function bathTrace(bounds){let pts=[];bathSegments(bounds).forEach(edge=>pts.push(...edge,[null,null,null]));let tr=pathTrace(pts,'Support-bath envelope');tr.line={width:2};return tr;}
function sceneExtent(r,amp){let max=0;r.vertices.forEach(p=>p.forEach(z=>max=Math.max(max,Math.abs(z))));
 if(r.context&&document.getElementById('context').value==='yes')r.context.bath_bounds.forEach(p=>p.forEach(z=>max=Math.max(max,Math.abs(z))));
 return max+Math.max(.8,amp*.65);}

function cpuDraw(){if(!cpuScene)return;let {r,vertices,amp}=cpuScene;
 const cy=Math.cos(cpu.yaw),sy=Math.sin(cpu.yaw),cp=Math.cos(cpu.pitch),sp=Math.sin(cpu.pitch);
 const rotate=p=>{let depth=-sy*p[0]+cy*p[1];return [cy*p[0]+sy*p[1],cp*p[2]-sp*depth,sp*p[2]+cp*depth]};
 let max=sceneExtent(r,amp);
 ids.forEach((id,idx)=>{let host=document.getElementById(id),canvas=host.querySelector('canvas.cpu');
  if(!canvas){host.innerHTML='';canvas=document.createElement('canvas');canvas.className='cpu';canvas.style.width='100%';canvas.style.height='100%';canvas.style.cursor='grab';host.appendChild(canvas);
   let last=null;canvas.addEventListener('pointerdown',e=>{last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});
   canvas.addEventListener('pointerup',()=>last=null);
   canvas.addEventListener('pointermove',e=>{if(!last)return;cpu.yaw+=(e.clientX-last[0])*.009;cpu.pitch=Math.max(-1.5,Math.min(1.5,cpu.pitch+(e.clientY-last[1])*.009));last=[e.clientX,e.clientY];cpuDraw()});
   canvas.addEventListener('wheel',e=>{e.preventDefault();cpu.zoom=Math.max(.4,Math.min(3,cpu.zoom*Math.exp(-e.deltaY*.001)));cpuDraw()},{passive:false});
  }
  const w=host.clientWidth,h=host.clientHeight,dpr=window.devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;
  let ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);let scale=Math.min(w,h)*.40/max*cpu.zoom;
  const vv=vertices[idx].map(rotate),project=p=>[w/2+scale*p[0],h/2-scale*p[1]];
  const faces=r.faces.map(f=>{let a=vv[f[0]],b=vv[f[1]],c=vv[f[2]],u=b.map((z,j)=>z-a[j]),v=c.map((z,j)=>z-a[j]);
   let n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],norm=Math.hypot(...n)||1;
   let light=Math.abs((n[0]*.25+n[1]*.45+n[2]*.86)/norm);return {f,depth:(a[2]+b[2]+c[2])/3,shade:Math.round(95+140*light)}}).sort((a,b)=>a.depth-b.depth);
  faces.forEach(({f,shade})=>{ctx.beginPath();f.forEach((i,j)=>{let p=project(vv[i]);j?ctx.lineTo(...p):ctx.moveTo(...p)});ctx.closePath();ctx.fillStyle=`rgb(${shade},${shade},${shade})`;ctx.fill()});
  if(idx>0&&document.getElementById('overlay').value==='yes'){let ref=r.vertices.map(rotate);ctx.globalAlpha=.12;ctx.lineWidth=.6;ctx.beginPath();r.faces.forEach(f=>{let p=project(ref[f[0]]);ctx.moveTo(...p);ctx.lineTo(...project(ref[f[1]]));ctx.lineTo(...project(ref[f[2]]));ctx.closePath()});ctx.stroke();ctx.globalAlpha=1;}
  if(r.context&&document.getElementById('context').value==='yes'){
   ctx.globalAlpha=.45;ctx.lineWidth=1;ctx.setLineDash([5,4]);ctx.beginPath();
   bathSegments(r.context.bath_bounds).forEach(edge=>{ctx.moveTo(...project(rotate(edge[0])));ctx.lineTo(...project(rotate(edge[1])))});ctx.stroke();ctx.setLineDash([]);
   const path=cpuScene.paths[idx];ctx.globalAlpha=.9;ctx.lineWidth=1.8;ctx.beginPath();path.forEach((p,j)=>{let v=project(rotate(p));j?ctx.lineTo(...v):ctx.moveTo(...v)});ctx.stroke();ctx.globalAlpha=1;
  }
  ctx.fillStyle='black';ctx.font='11px system-ui';ctx.fillText('Drag to rotate · scroll to zoom · CPU mesh renderer',14,h-14);
  [[1,0,0],[0,1,0],[0,0,1]].forEach((p,i)=>{let a=rotate(p),x=40+27*a[0],y=45-27*a[1];ctx.beginPath();ctx.moveTo(40,45);ctx.lineTo(x,y);ctx.stroke();ctx.fillText(['x','y','z'][i],x+3,y-3)});
 });
}
window.addEventListener('resize',()=>{if(!USE_WEBGL)cpuDraw()});
const byId=id=>document.getElementById(id);
DATA.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.process.toUpperCase()+' / '+r.shape.replaceAll('_',' ');byId('case').appendChild(o)});
const initialCase=DATA.findIndex(r=>r.name===location.hash.slice(1));const embeddedCase=DATA.findIndex(r=>r.name==='embedded_helix');byId('case').value=String(initialCase>=0?initialCase:(embeddedCase>=0?embeddedCase:0));
for(let i=0;i<DATA[0].samples0.length;i++){let o=document.createElement('option');o.value=i;o.textContent='Paired realization '+(i+1);byId('realization').appendChild(o)}
function shifted(v,d,s){return v.map((p,i)=>p.map((z,j)=>z+s*d[i][j]));}
function meshTrace(r,v,name,opacity=1){return {type:'mesh3d',x:v.map(p=>p[0]),y:v.map(p=>p[1]),z:v.map(p=>p[2]),
 i:r.faces.map(f=>f[0]),j:r.faces.map(f=>f[1]),k:r.faces.map(f=>f[2]),name:name,opacity:opacity,
 flatshading:false,lighting:{ambient:.45,diffuse:.8,specular:.25,roughness:.6},
 hovertemplate:'x %{x:.3f} mm<br>y %{y:.3f} mm<br>z %{z:.3f} mm<extra>'+name+'</extra>'};}
function layout(r,amp){let max=sceneExtent(r,amp);
return {margin:{l:0,r:0,t:0,b:0},showlegend:false,uirevision:r.name,
scene:{aspectmode:'cube',camera:camera,xaxis:{title:'x / mm',range:[-max,max]},yaxis:{title:'y / mm',range:[-max,max]},zaxis:{title:'z / mm',range:[-max,max]}}};}
function pct(p){return (100*p).toFixed(2)+'%'}
function ci(v){const digits=(v.estimate===0||v.estimate===1)?5:2;const fmt=p=>(100*p).toFixed(digits)+'%';return '95% interval: '+fmt(v.ci_lower)+'–'+fmt(v.ci_upper)}
async function render(){const r=DATA[+byId('case').value],amp=+byId('scale').value,sel=byId('realization').value;
byId('description').textContent=r.description;
byId('context').disabled=!r.context;
byId('command-caption').textContent=r.context?'Nozzle centerline and equivalent-radius command envelope — not the predicted final filament.':'Pre-distorted design surface — not the expected finished part.';
byId('geometry-note').textContent=r.context?'Embedded extrusion uses separate centerline and radius budgets that sum to the displayed surface tolerance. Exact budgets are in embedded_model.json. Bath outlines are context, not simulated fluid. Target reference frames stay fixed.':'Surface-normal offsets use a nonnegative partition-of-unity basis.';
byId('required').textContent=pct(r.target_probability);
byId('box').textContent=pct(r.validation.optimized_box.estimate);byId('box-ci').textContent=ci(r.validation.optimized_box);
byId('mesh').textContent=pct(r.validation.optimized_mesh.estimate);byId('mesh-ci').textContent=ci(r.validation.optimized_mesh);
byId('tol').textContent='≤ '+r.tolerance_mm.toFixed(3)+' mm';
byId('status').textContent=(amp===1?'True-scale geometry.':'DISPLAY ONLY: deformations amplified '+amp+'×; metrics are unamplified. Large amplification can invert displayed radii; use 1× for physical geometry.')+
' '+r.status+' after '+r.iterations+' LP iterations. Working coefficient-box target '+pct(r.working_probability)+'.';
const d0=sel==='mean'?r.mean0:r.samples0[+sel],d1=sel==='mean'?r.mean1:r.samples1[+sel];
byId('baseline-caption').textContent=(sel==='mean'?'Mean prediction.':'Same random manufacturing draw as the compensated panel.')+' Baseline mesh yield '+pct(r.validation.baseline_mesh.estimate)+'.';
byId('optimized-caption').textContent=(sel==='mean'?'Mean prediction.':'Compensation applied to the paired random draw.')+' '+(amp===1?'True-scale.':amp+'× display deformation.');
const vertices=[r.vertices,shifted(r.vertices,r.command,amp),shifted(r.vertices,d0,amp),shifted(r.vertices,d1,amp)];
let paths=null;
if(r.context){const c=r.context,t=c.target_path;
 const amplify=p=>t.map((v,i)=>v.map((z,j)=>z+amp*(p[i][j]-z)));
 paths=[t,amplify(c.command_path),amplify(sel==='mean'?c.mean0_path:c.samples0_path[+sel]),amplify(sel==='mean'?c.mean1_path:c.samples1_path[+sel])];}
if(!USE_WEBGL){cpuScene={r,vertices,amp,paths};cpuDraw();}else await Promise.all(ids.map((id,i)=>{let traces=[meshTrace(r,vertices[i],id)];if(i>0&&byId('overlay').value==='yes')traces.push(meshTrace(r,r.vertices,'target reference',.20));
if(r.context&&byId('context').value==='yes'){traces.push(bathTrace(r.context.bath_bounds));traces.push(pathTrace(paths[i],'Nozzle / filament centerline'));}
return Plotly.react(id,traces,layout(r,amp),{responsive:true,displaylogo:false});}));
Plotly.react('convergence',[{x:r.history.map(h=>h.iteration),y:r.history.map(h=>h.upper_bound),name:'Feasible incumbent cost',mode:'lines'},
{x:r.history.map(h=>h.iteration),y:r.history.map(h=>h.lower_bound),name:'LP relaxation cost',mode:'lines'}],
{margin:{l:65,r:20,t:15,b:45},xaxis:{title:'LP iteration'},yaxis:{title:'Dimensionless compensation cost'},legend:{orientation:'h'}},{responsive:true,displaylogo:false});}
['case','realization','scale','overlay','context'].forEach(id=>byId(id).addEventListener('change',render));
render().then(()=>{if(USE_WEBGL)ids.forEach(id=>byId(id).on('plotly_relayout',e=>{if(lock||!e['scene.camera'])return;camera=e['scene.camera'];lock=true;
Promise.all(ids.filter(x=>x!==id).map(x=>Plotly.relayout(x,{'scene.camera':camera}))).finally(()=>{lock=false});}));});
</script></body></html>"""
