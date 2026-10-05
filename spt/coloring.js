// Every channel increases from low to high, so relative luminance is monotonic.
// Keep the high end visible on the workspace's white background.
export const numericPalettes = {
  blue: ['#122e60', '#84c3f4'],
  teal: ['#08413e', '#74dab6'],
  amber: ['#582d0a', '#f8c652'],
  grayscale: ['#2d2d2d', '#c3c3c3'],
};
export const categoricalPalettes = {
  balanced: ['#3568c0','#d95f30','#279572','#9a62b0','#b78c22','#2594b2','#cc568d','#667849','#786556','#817fc2','#ca993e','#47766e'],
  bright: ['#1565c0','#e65100','#00875a','#8e24aa','#c62828','#00838f','#ad7b00','#d81b60','#5e35b1','#558b2f','#795548','#455a64'],
  muted: ['#547a9e','#b57b58','#659079','#9275a4','#b26879','#55949a','#a69859','#778466','#997e72','#7b80ad','#aa8a6c','#6e8d8b'],
};

export function createColoring(spec, knownIds) {
  if (!spec) return null;
  const known = new Set(knownIds), values = Object.entries(spec.values || {});
  if (!values.length || values.some(([id])=>!known.has(id))) {
    const error = new Error('Color mapping includes unknown trajectories.');error.code='unknown_track';throw error;
  }
  let entries, color, ramp=null;
  if (spec.kind==='numeric') {
    if(values.some(([,v])=>!Number.isFinite(v)))throw new Error('Invalid numeric color value.');
    const numbers=values.map(([,v])=>v),min=Math.min(...numbers),max=Math.max(...numbers);
    const paletteName=spec.palette??'blue';
    if(!Object.hasOwn(numericPalettes,paletteName))throw new Error('Unsupported numeric palette.');
    spec={...spec,palette:paletteName};
    const endpoints=numericPalettes[paletteName];
    const stops=endpoints.map(hex=>[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)));
    color=v=>{const t=max===min?.5:(v-min)/(max-min);
      return '#'+stops[0].map((n,i)=>Math.round(n+(stops[1][i]-n)*t).toString(16).padStart(2,'0')).join('');};
    ramp=max===min?[color(min),color(min)]:endpoints;
    const endpointLabel=(value,digits)=>{
      const rounded=Number(value.toPrecision(digits));
      return String(Number.isFinite(rounded)?rounded:value);
    };
    let digits=4,labels=[endpointLabel(min,digits),endpointLabel(max,digits)];
    // Preserve a readable range even when four significant digits collapse it.
    while(min!==max&&labels[0]===labels[1]&&digits<17){
      digits++;labels=[endpointLabel(min,digits),endpointLabel(max,digits)];
    }
    entries=[{label:labels[0],color:color(min)},{label:labels[1],color:color(max)}];
  } else if (spec.kind==='categorical') {
    const paletteName=spec.palette??'balanced';
    if(!Object.hasOwn(categoricalPalettes,paletteName))throw new Error('Choose balanced, bright, or muted for categories.');
    spec={...spec,palette:paletteName};
    const palette=categoricalPalettes[paletteName];
    const groups=[...new Set(values.map(([,v])=>v))].sort();
    if(groups.length>12||groups.some(v=>typeof v!=='string'))throw new Error('Invalid category labels.');
    const colors=new Map(groups.map((g,i)=>[g,palette[i]]));color=v=>colors.get(v);
    entries=groups.map(label=>({label,color:color(label)}));
  } else throw new Error('Unsupported coloring.');
  return {spec,entries,ramp,colors:new Map(values.map(([id,value])=>[id,color(value)])),mappedCount:values.length,missingCount:known.size-values.length};
}
