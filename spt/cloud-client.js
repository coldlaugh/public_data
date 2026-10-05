// Cloud-only transport. The local prototype remains unchanged.
export function createCloudApi({origin,fetchImpl=fetch,encode,now=()=>Date.now(),onExpiry=()=>{},onUploadProgress=()=>{},clientId}){
  const objects=new Map();
  if(!clientId){try{clientId=localStorage.getItem('spt.browser.identity');}catch{}}
  if(!/^[0-9a-f]{32}$/.test(clientId||'')){clientId=crypto.randomUUID().replaceAll('-','');try{localStorage.setItem('spt.browser.identity',clientId);}catch{}}
  const digest=async bytes=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
  const decode=data=>{const text=atob(data),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return bytes;};
  async function wire(path,data,auth){
    const response=await fetchImpl(new URL(path,origin),{method:data===undefined?'GET':'POST',headers:{'Content-Type':'application/json',...(auth?{Authorization:'Bearer '+auth.token}:{})},body:data===undefined?undefined:JSON.stringify(data)});
    const result=await response.json();
    if(!response.ok){if(response.status===410){objects.clear();onExpiry(auth?.id);}throw new Error(result.error||'Request failed.');}
    return result;
  }
  async function retry(path,data,auth){
    let error;for(let i=0;i<3;i++){try{return await wire(path,data,auth);}catch(e){error=e;if(i<2)await new Promise(resolve=>setTimeout(resolve,500*(i+1)));}}throw error;
  }
  async function storedObject(path,auth){
    let response;
    for(let attempt=0;attempt<3;attempt++){
      try {
        response=await fetchImpl(new URL(path,origin),{headers:{Authorization:'Bearer '+auth.token}});
      } catch(error){
        if(attempt===2)throw error;
        await new Promise(resolve=>setTimeout(resolve,500*(attempt+1)));continue;
      }
      if(response.ok)return response;
      if(response.status===410){objects.clear();onExpiry(auth?.id);throw new Error('This session expired after seven days.');}
      if(![404,409,429,500,502,503,504].includes(response.status)||attempt===2)break;
      await new Promise(resolve=>setTimeout(resolve,500*(attempt+1)));
    }
    throw new Error('Unable to load stored artifact.');
  }
  async function upload(path,data,auth){
    const metadata=[];let total=0;
    for(const f of data.files){const raw=decode(f.data);total+=raw.length;metadata.push({name:f.name,bytes:raw.length,sha256:await digest(raw)});}
    const prepared=await retry(path==='/api/jobs'?'/api/jobs/prepare':path.replace(/messages$/,'prepare'),{...data,files:metadata,clientId},auth);
    if(prepared.complete)return prepared;
    const credential=auth||prepared;let sent=0;
    for(let file=0;file<data.files.length;file++){
      const raw=decode(data.files[file].data),count=Math.max(1,Math.ceil(raw.length/prepared.chunkBytes));
      for(let part=0;part<count;part++){
        const chunk=raw.subarray(part*prepared.chunkBytes,(part+1)*prepared.chunkBytes);
        if(!prepared.received[file][part])await retry(`/api/jobs/${prepared.id}/upload-chunk`,{uploadId:prepared.uploadId,file,part,data:encode(chunk)},credential);
        sent+=chunk.length;onUploadProgress(sent,total);
      }
    }
    return retry(`/api/jobs/${prepared.id}/upload-finish`,{uploadId:prepared.uploadId},credential);
  }
  async function request(path,data,auth){
    if(data&&(path==='/api/jobs'||/\/messages$/.test(path))){
      data={...data,clientId};
      if(data.files?.length)return upload(path,data,auth);
    }
    const result=await wire(path,data,auth);
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
              const parts=file.chunks||[file.objectRef],buffers=[];let length=0;
              for(const part of parts){
                if(!/^[0-9a-f]{64}$/.test(part))throw new Error('Invalid chunk reference.');
                const r=await storedObject(`/api/jobs/${result.id}/objects/${part}`,credential);
                const bytes=new Uint8Array(await r.arrayBuffer());
                if(await digest(bytes)!==part)throw new Error('Stored chunk checksum mismatch.');
                length+=bytes.length;if(length>file.bytes)throw new Error('Stored file size mismatch.');buffers.push(bytes);
              }
              const bytes=new Uint8Array(length);let offset=0;for(const part of buffers){bytes.set(part,offset);offset+=part.length;}
              if(await digest(bytes)!==file.objectRef||length!==file.bytes)throw new Error('Stored artifact checksum mismatch.');
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
