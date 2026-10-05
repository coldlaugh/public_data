// Keep historical response links intact while identifying superseded file contents.
export function artifactVersion(file,artifacts){
  const latest=artifacts.filter(item=>item.name===file.name).at(-1)||file;
  const sameContent=latest===file || (file.sha256&&latest.sha256===file.sha256) ||
    (file.data!==undefined&&latest.data===file.data);
  return {response:(file.turn??0)+1,superseded:!sameContent,latest};
}

// Resolve only artifacts already attached to this conversation; never fetch local paths.
export function resolveArtifactLink(href,artifacts,turn=Infinity){
  if(typeof href!=='string')return null;
  href=href.trim();if(/^https?:|^\/\//i.test(href))return null;
  let path;try{path=decodeURIComponent(href).replace(/^sandbox:/i,'').replace(/^file:\/\//i,'').split(/[?#]/)[0].replaceAll('\\','/');}catch{return null;}
  if(/^[a-z][a-z0-9+.-]*:/i.test(path))return null;
  const files=artifacts.filter(f=>(f.turn??0)<=turn);
  const candidates=new Set([path.replace(/^\.\//,''),path.replace(/^.*(?:^|\/)outputs\//,''),path.replace(/^\/mnt\/data\//,'')]);
  for(const candidate of candidates){const matches=files.filter(f=>f.name===candidate||f.name===candidate.replaceAll('/','__'));if(matches.length)return matches.at(-1);}
  const basename=path.split('/').at(-1),matches=files.filter(f=>f.name.split('/').at(-1)===basename);
  const names=new Set(matches.map(f=>f.name));return names.size===1?matches.at(-1):null;
}
