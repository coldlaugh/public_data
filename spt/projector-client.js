// An explicit analysis link must never fall back to a different saved dataset.
export function selectSavedJob(saved,fragment){
  if(fragment.has('job'))return saved.find(job=>job.id===fragment.get('job'))||null;
  return fragment.get('new')==='1'?null:saved.at(-1)||null;
}

// Switching views must preserve an explicit selection, including IDs currently
// hidden by isolation. A raw-only selection cannot be silently widened to all.
export function planProjectionSelection(ids,availableIds,visibleIds){
  const available=new Set(availableIds),visible=visibleIds==null?null:new Set(visibleIds);
  return {ids:[...ids],unavailable:ids.filter(id=>!available.has(id)),restoreAll:!!visible&&ids.some(id=>!visible.has(id))};
}

// Pick in screen pixels, including between samples on a raw trajectory.
export function pickTrajectory(tracks,[x,y],radius=8){
  let nearest=null,best=radius*radius;
  for(const track of tracks){
    for(let i=0;i<track.path.length;i++){
      const a=track.path[i],b=track.path[i+1]||a,dx=b[0]-a[0],dy=b[1]-a[1];
      const t=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy||1)));
      const distance=(x-a[0]-t*dx)**2+(y-a[1]-t*dy)**2;
      if(distance<=best){nearest=track.key;best=distance;}
    }
  }
  return nearest;
}

export function questionForDisplay(turn){
  let text=turn.question,selection=turn.selection;
  if(!selection){
    const legacy=/\n\nCurrent trajectory selection in the browser: [\s\S]+\. These identify file names and track IDs, not an independent dataset\.$/;
    if(legacy.test(text)){text=text.replace(legacy,'');selection={legacy:true};}
  }
  return {text:text.split('\n\nDemo dataset context:\n')[0],selection};
}

// Browser-side, job-scoped projector RPC. The adapter can later wrap the legacy UI.
export class ProjectorClient {
  constructor({api, context, state, project, select, color, storage, clientId}) {
    Object.assign(this, {api, context, state, project, select, color, storage, clientId});
    this.busy = false;
    this.cache = new Map();
  }
  async tick(job, auth) {
    if (this.busy || this.context()?.id !== job.id) return;
    const action = job.projectorActions?.find(a => a.expires * 1000 > Date.now());
    if (!action) return;
    this.busy = true;
    const path = `/api/jobs/${job.id}/projector-actions/${action.id}`;
    const cacheKey = `spt.projector.result.${job.id}.${action.id}`;
    try {
      const claim = await this.api(path + '/claim', {clientId:this.clientId}, auth);
      if (!claim.claimed) return;
      let result = this.cache.get(cacheKey);
      if (!result) {
        try { result = JSON.parse(this.storage.getItem(cacheKey) || 'null'); } catch {}
      }
      if (!result) {
        result = await this.apply(claim, job.id);
        this.cache.set(cacheKey, result);
        try { this.storage.setItem(cacheKey, JSON.stringify(result)); } catch {}
      }
      await this.api(path + '/result', {clientId:this.clientId, result}, auth);
    } finally { this.busy = false; }
  }
  async apply(action, jobId) {
    if (this.context()?.id !== jobId) return {ok:false,error:'view_changed'};
    if (Date.now() >= action.expires * 1000) return {ok:false,error:'timeout'};
    try {
      const request = action.request;
      if (request.op === 'wait_embeddings') {
        while(true){
          if(this.context()?.id!==jobId)return {ok:false,error:'view_changed'};
          const state=this.state();
          if(state.embeddingPublications?.includes(request.publicationId)&&state.projectionReady&&state.projection!=='raw')break;
          if(Date.now()>=action.expires*1000)return {ok:false,error:'timeout'};
          await new Promise(resolve=>setTimeout(resolve,200));
        }
      } else if (request.op === 'set_projection') {
        const result = await this.project(request.method);
        if (result?.cancelled) return {ok:false,error:'view_changed'};
      } else if (request.op === 'select_tracks') {
        this.select(request.ids);
      } else if (request.op === 'color_tracks' || request.op === 'clear_colors') {
        this.color(request.coloring || null);
      } else if (request.op !== 'get_state') return {ok:false,error:'unavailable'};
      if (this.context()?.id !== jobId) return {ok:false,error:'view_changed'};
      if (Date.now() >= action.expires * 1000) return {ok:false,error:'timeout'};
      return {ok:true, ...this.state()};
    } catch (error) {
      return {ok:false,error:error?.code === 'unknown_track' ? 'unknown_track' : 'projection_failed'};
    }
  }
}

// Accept only complete publications; a partial retry must not remove old points.
export function readEmbeddings(artifacts, decode) {
  const groups=new Map(), legacy=[];
  for(const [order,file] of artifacts.entries()){
    const p=JSON.parse(decode(file.data));
    if(p.schema==='spt.encoder.v1'){legacy.push({...p,artifactId:file.id||file.name});continue;}
    if(p.schema!=='spt.encoder.v2'||!Array.isArray(p.points))throw new Error('Unsupported encoder vector format.');
    const pub=p.publication;
    if(!pub||!Number.isInteger(pub.parts)||pub.parts<1||pub.parts>20000||!Number.isInteger(pub.part)||pub.part<0||pub.part>=pub.parts||!Array.isArray(pub.sources))throw new Error('Invalid embedding publication.');
    if(!groups.has(pub.id))groups.set(pub.id,{parts:new Map(),order,pub});
    const group=groups.get(pub.id);
    if(group.pub.parts!==pub.parts||group.pub.totalPoints!==pub.totalPoints||JSON.stringify(group.pub.sources)!==JSON.stringify(pub.sources))throw new Error('Inconsistent embedding publication.');
    group.parts.set(pub.part,p);group.order=order;
  }
  const byId=new Map(),provenances=[],sourceOwners=new Map();
  for(const p of legacy){for(const point of p.points||[])byId.set(point.id,point);provenances.push(p.provenance);}
  for(const group of [...groups.values()].sort((a,b)=>a.order-b.order)){
    if(group.parts.size!==group.pub.parts)continue;
    const decoded=[];
    for(let n=0;n<group.pub.parts;n++){
      const p=group.parts.get(n),v=p.vectors;
      if(v?.encoding!=='float32-le/base64'||v.dimensions!==512)throw new Error('Invalid encoder dimensions.');
      const bytes=Uint8Array.from(atob(v.data),c=>c.charCodeAt(0));
      if(bytes.length!==p.points.length*512*4)throw new Error('Incomplete encoder vectors.');
      const view=new DataView(bytes.buffer);
      for(let i=0;i<p.points.length;i++){
        const point=p.points[i],vector=Array.from({length:512},(_,d)=>view.getFloat32((i*512+d)*4,true));
        if(!group.pub.sources.includes(point.file)||point.id!==point.file+':'+point.trackId||vector.some(x=>!Number.isFinite(x)))throw new Error('Invalid encoder point.');
        decoded.push({...point,vector});
      }
    }
    if(decoded.length!==group.pub.totalPoints||new Set(decoded.map(p=>p.id)).size!==decoded.length)throw new Error('Incomplete embedding publication.');
    for(const [id,p] of byId)if(group.pub.sources.includes(p.file))byId.delete(id);
    for(const point of decoded)byId.set(point.id,point);
    provenances.push(group.parts.get(0).provenance);
    for(const source of group.pub.sources)sourceOwners.set(source,group.pub.id);
  }
  const publications=[...new Set(sourceOwners.values())];
  return {schema:'spt.encoder.v1',points:[...byId.values()],provenances,publications,authoritativeSources:[...sourceOwners.keys()],version:publications.join(':')+':'+legacy.map(p=>p.artifactId).join(':')};
}
