// Cloud-only transport. The local prototype remains unchanged.
export function createCloudApi({origin,fetchImpl=fetch,encode,now=()=>Date.now(),onExpiry=()=>{}}){
  const objects=new Map();
  async function request(path,data,auth){
    const response=await fetchImpl(new URL(path,origin),{method:data===undefined?'GET':'POST',
      headers:{'Content-Type':'application/json',...(auth?{Authorization:'Bearer '+auth.token}:{})},
      body:data===undefined?undefined:JSON.stringify(data)});
    const result=await response.json();
    if(!response.ok){if(response.status===410){objects.clear();onExpiry(auth?.id);}
      throw new Error(result.error||'Request failed.');}
    if(result.expiresAt&&result.expiresAt*1000<=now()){objects.clear();onExpiry(result.id);throw new Error('This session expired after seven days.');}
    if(result.id&&result.files){
      const credential=auth||result;
      const current=new Set();
      for(const field of ['files','artifacts']){
        for(const file of result[field]||[]){
          if(!file.objectRef||file.data!==undefined)continue;
          if(!/^[0-9a-f]{64}$/.test(file.objectRef))throw new Error('Invalid stored artifact reference.');
          const key=result.id+':'+file.objectRef;current.add(key);
          if(!objects.has(key)){
            const pending=(async()=>{
              const r=await fetchImpl(new URL(`/api/jobs/${result.id}/objects/${file.objectRef}`,origin),{headers:{Authorization:'Bearer '+credential.token}});
              if(!r.ok)throw new Error('Unable to load stored artifact.');
              const bytes=new Uint8Array(await r.arrayBuffer());
              const actual=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
              if(actual!==file.objectRef||bytes.length!==file.bytes)throw new Error('Stored artifact checksum mismatch.');
              return encode(bytes);
            })();
            objects.set(key,pending);pending.catch(()=>objects.delete(key));
          }
          file.data=await objects.get(key);
        }
      }
      for(const key of objects.keys())if(!current.has(key))objects.delete(key);
    }
    return result;
  }
  return request;
}

export function cloudPollDelay(job,hidden){
  if(hidden)return 30000;
  return job&&['running','queued','cancelling'].includes(job.status)?2500:60000;
}
