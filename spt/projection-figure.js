// Export the rendered camera view with the same scientific context as the workspace.
export function projectionFigureContext({snapshot,settings,coloring,legendNote,filenames}){
  const method=String(snapshot.projection).toUpperCase();
  const is3D=snapshot.projection==='umap'?settings?.umapIs3d:snapshot.projection==='tsne'?settings?.tSNEis3d:settings?.pcaIs3d;
  return {
    title:(snapshot.projection==='pca'?'Uncentered PCA':method)+' encoder projection'+(typeof is3D==='boolean'?' · '+(is3D?'3D':'2D'):''),
    scope:snapshot.pointCount+' trajector'+(snapshot.pointCount===1?'y':'ies')+' in this projection · '+snapshot.selectedCount+' selected',
    datasets:'Datasets: '+filenames.join(', '),
    explanation:'One point per trajectory. Projection distances have no spatial units; clusters alone do not establish a motion model.',
    rerun:snapshot.projection==='umap'||snapshot.projection==='tsne'?'Stochastic projection; rerunning can change the layout.':'Uncentered SVD of the current trajectory group; component shares describe squared vector magnitude, including the mean. Centered variance is not reported. Random dimension reduction or sampling can change the result.',
    legend:coloring?{title:coloring.spec.label+(coloring.spec.units?' ('+coloring.spec.units+')':''),kind:coloring.spec.kind,entries:coloring.entries.map(e=>({...e})),ramp:coloring.ramp?[...coloring.ramp]:null,note:legendNote}:null,
    selectionNote:coloring?'Selected points are enlarged and labelled; colors retain supplied values.':'Selected points are enlarged and labelled using the native projector display colors.',
  };
}

export async function renderProjectionFigure(plot,context){
  if(!plot?.width||!plot?.height)throw new Error('The projection is not ready to export.');
  const width=Math.max(960,Math.min(4096,plot.width)),plotHeight=Math.round(plot.height*width/plot.width);
  const canvas=document.createElement('canvas'),probe=canvas.getContext('2d');
  const font=Math.max(18,Math.round(width/75)),pad=font*1.5,line=font*1.45;
  const blocks=[];let footerHeight=pad;
  // Break long identifiers as well as ordinary prose; preserve every character.
  const text=(value,bold=false)=>{
    probe.font=(bold?'600 ':'')+font+'px Arial, sans-serif';
    let row='';
    for(const char of String(value)){
      if(char==='\n'||probe.measureText(row+char).width>width-2*pad){
        const split=char==='\n'?-1:row.lastIndexOf(' ');
        blocks.push({text:split>0?row.slice(0,split):row,bold,y:footerHeight});footerHeight+=line;
        row=char==='\n'?'':(split>0?row.slice(split+1):'')+char;
      }
      else row+=char;
    }
    blocks.push({text:row,bold,y:footerHeight});footerHeight+=line;
  };
  text(context.title,true);text(context.scope);text(context.datasets);
  if(context.legend){
    footerHeight+=font*.5;text(context.legend.title,true);
    if(context.legend.ramp){blocks.push({ramp:context.legend.ramp,y:footerHeight});footerHeight+=font*1.7;text(context.legend.entries.map(e=>e.label).join(' → '));}
    else for(const entry of context.legend.entries){blocks.push({swatch:entry.color,y:footerHeight});text('    '+entry.label);}
    text(context.legend.note);
  }else text(context.selectionNote);
  footerHeight+=font*.5;text(context.explanation);text(context.rerun);
  canvas.width=width;canvas.height=plotHeight+Math.ceil(footerHeight+pad);
  const ctx=canvas.getContext('2d');ctx.fillStyle='#ffffff';ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(plot,0,0,width,plotHeight);
  ctx.strokeStyle='#dce3ef';ctx.beginPath();ctx.moveTo(0,plotHeight);ctx.lineTo(width,plotHeight);ctx.stroke();
  for(const block of blocks){
    const y=plotHeight+block.y;
    if(block.ramp){const gradient=ctx.createLinearGradient(pad,0,width-pad,0);block.ramp.forEach((color,i)=>gradient.addColorStop(i/(block.ramp.length-1),color));ctx.fillStyle=gradient;ctx.fillRect(pad,y,width-2*pad,font*.7);}
    else if(block.swatch){ctx.fillStyle=block.swatch;ctx.fillRect(pad,y-font*.8,font*.7,font*.7);}
    else{ctx.fillStyle='#26354b';ctx.font=(block.bold?'600 ':'')+font+'px Arial, sans-serif';ctx.fillText(block.text,pad,y);}
  }
  const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));
  if(!blob)throw new Error('The projection figure could not be exported.');
  return blob;
}
