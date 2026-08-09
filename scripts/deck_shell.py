"""HTML shell for the presentation: layout CSS + navigation JS.

Kept apart from build_presentation.py so the slide content stays readable. Placeholders
__AUTHOR__, __SLIDES__ and __TITLES__ are filled by the builder.
"""

SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<!-- Without this, Chrome's auto-dark-mode inverts the palette into dark-on-dark. -->
<meta name="color-scheme" content="light only">
<title>Tomato Leaf Disease Detector — __AUTHOR__</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Work+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  color-scheme:light only;
  --bg:#EEF1E7; --surface:#FFFFFF; --ink:#1F2A24;
  --soft:rgba(31,42,36,.86); --mute:rgba(31,42,36,.58); --faint:rgba(31,42,36,.42);
  --brand:#1E4635; --accent:#B8863B; --line:rgba(30,70,53,.14); --line2:rgba(30,70,53,.07);
  --good:#3F6B52;
  --serif:'Fraunces',Georgia,'Times New Roman',serif;
  --sans:'Work Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,'SF Mono',Menlo,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#0d120f;font-family:var(--sans);color:var(--ink);overflow:hidden}
#stage{position:fixed;inset:0;display:grid;place-items:center}
/* Fixed 1280x720 canvas scaled to fit -- layout is identical on every screen, so nothing
   reflows unexpectedly mid-presentation. */
#deck{width:1280px;height:720px;position:relative;transform-origin:center center;
  box-shadow:0 24px 80px rgba(0,0,0,.55);border-radius:6px;overflow:hidden}
.slide{position:absolute;inset:0;background:var(--bg);padding:44px 58px 52px;
  display:none;flex-direction:column;animation:in .3s ease both}
.slide.on{display:flex}
@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.eyebrow{font-size:11px;font-weight:600;letter-spacing:.2em;text-transform:uppercase;
  color:var(--accent);margin-bottom:9px}
h1{font-family:var(--serif);font-weight:600;font-size:48px;line-height:1.07;color:var(--brand);
  letter-spacing:-.02em}
h2{font-family:var(--serif);font-weight:600;font-size:31px;line-height:1.15;color:var(--brand);
  letter-spacing:-.015em;margin-bottom:4px}
h3{font-family:var(--serif);font-weight:600;font-size:16.5px;color:var(--brand);margin-bottom:5px}
p{font-size:15px;line-height:1.58;color:var(--soft)}
.sub{font-size:17px;color:var(--mute);margin-top:10px;max-width:940px}
strong{color:var(--brand);font-weight:600}
code{font-family:var(--mono);font-size:.87em;background:rgba(30,70,53,.08);
  padding:1px 5px;border-radius:4px;color:var(--brand)}
.body{flex:1;min-height:0;display:flex;flex-direction:column;gap:11px;margin-top:3px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;flex:1;min-height:0;
  align-content:stretch}
/* Columns inside .two spread their cards over the full height, so a slide never leaves a
   block of dead space below the content. */
.two>div{display:flex;flex-direction:column;gap:11px}
.two>div>.card{flex:1 1 auto;display:flex;flex-direction:column;justify-content:center}
.two>div>.card>ul{flex:0 1 auto}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}
.card{background:var(--surface);border:1px solid var(--line2);border-radius:12px;
  padding:13px 15px;box-shadow:0 2px 12px rgba(30,70,53,.06)}
.card.lead{border-left:3px solid var(--accent)}
.card.good{border-left:3px solid var(--good)}
ul{list-style:none;display:flex;flex-direction:column;gap:7px}
li{font-size:14px;line-height:1.5;color:var(--soft);padding-left:16px;position:relative}
li::before{content:'';position:absolute;left:0;top:.55em;width:5px;height:5px;border-radius:50%;
  background:var(--accent)}
.num{font-family:var(--serif);font-size:40px;font-weight:600;color:var(--brand);line-height:1;
  font-variant-numeric:tabular-nums}
.num.sm{font-size:28px}
.lab{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--mute);
  font-weight:600;margin-bottom:5px}
.foot{font-size:11.5px;color:var(--faint);margin-top:6px;line-height:1.5}
.row{display:flex;gap:11px}
.row>*{flex:1}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--mute);
  font-weight:600;text-align:right;padding:4px 6px;border-bottom:1px solid var(--line)}
th:first-child{text-align:left}
td{padding:3.5px 6px;text-align:right;border-bottom:1px solid var(--line2);
  font-variant-numeric:tabular-nums;color:var(--soft)}
td:first-child{text-align:left;font-weight:500;color:var(--ink)}
tr.hi td{background:rgba(184,134,59,.11)}
tr.hi td:first-child{color:var(--accent);font-weight:600}
figure{flex:1;min-height:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:7px}
figure img{max-width:100%;max-height:100%;object-fit:contain;border-radius:8px;
  background:#fff;border:1px solid var(--line2)}
figcaption{font-size:11.5px;color:var(--faint);text-align:center;max-width:94%;line-height:1.5}
.bar{height:7px;border-radius:4px;background:rgba(30,70,53,.1);overflow:hidden;margin-top:5px}
.bar>i{display:block;height:100%;border-radius:4px;
  background:linear-gradient(90deg,var(--accent),var(--brand))}
.pill{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.06em;
  padding:3px 9px;border-radius:20px;background:rgba(30,70,53,.1);color:var(--brand)}
.flow{display:flex;align-items:stretch;gap:5px;flex:0 0 auto}
.fs{flex:1;background:var(--surface);border:1px solid var(--line2);border-radius:10px;
  padding:9px 10px;box-shadow:0 2px 10px rgba(30,70,53,.05)}
.fs.on{border-left:3px solid var(--accent)}
.fs b{font-family:var(--serif);font-size:13.5px;color:var(--brand);display:block;
  margin-bottom:3px;font-weight:600}
.fs span{font-size:11px;color:var(--mute);line-height:1.42;display:block}
.fa{display:grid;place-items:center;color:var(--accent);font-size:16px;flex:0 0 auto}
.slide.dark{background:var(--brand)}
.slide.dark h1,.slide.dark h2,.slide.dark h3,.slide.dark strong{color:#EEF1E7}
.slide.dark p,.slide.dark .sub{color:rgba(238,241,231,.78)}
.slide.dark .eyebrow{color:#D9A85C}
.slide.dark .num{color:#EEF1E7}
.slide.dark .lab{color:rgba(238,241,231,.6)}
.slide.dark .card{background:rgba(255,255,255,.06);border-color:rgba(238,241,231,.14);
  box-shadow:none}
.slide.dark li{color:rgba(238,241,231,.82)}
.slide.dark code{background:rgba(255,255,255,.12);color:#cfe3d5}
.cm{border-collapse:separate;border-spacing:2px;font-size:10.5px;width:auto;margin:0 auto}
.cm td,.cm th{border:0;padding:0}
.cm .rl{text-align:right;padding-right:8px;color:var(--soft);white-space:nowrap;font-weight:500}
.cm .cl{writing-mode:vertical-rl;transform:rotate(180deg);font-size:10px;color:var(--mute);
  height:90px;padding-bottom:5px;font-weight:500}
.cm .c{width:33px;height:24px;border-radius:3px;text-align:center;vertical-align:middle;
  font-variant-numeric:tabular-nums;font-size:10.5px;cursor:default}
.cm .c.d{outline:1.5px solid var(--accent);outline-offset:-1.5px}
.brand-mark{display:flex;align-items:center;gap:14px;margin-bottom:20px}
.brand-mark .m{width:50px;height:50px;border-radius:14px;background:var(--brand);
  display:grid;place-items:center;font-size:25px}
.brand-mark .w{font-family:var(--serif);font-size:19px;font-weight:600;color:var(--brand)}
.brand-mark .t{font-size:10px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--faint);font-weight:600;margin-top:2px}
.author{margin-top:auto;padding-top:18px;border-top:1px solid var(--line);display:flex;
  align-items:flex-end;justify-content:space-between;gap:20px}
.author .nm{font-family:var(--serif);font-size:25px;font-weight:600;color:var(--brand)}
.author .mt{font-size:12.5px;color:var(--mute);margin-top:4px}
.headline{display:flex;gap:30px;margin-top:18px}
#bar{position:fixed;top:0;left:0;height:3px;background:var(--accent);z-index:20;
  transition:width .25s ease}
#hud{position:fixed;bottom:14px;right:18px;font-family:var(--mono);font-size:11.5px;
  color:rgba(238,241,231,.5);z-index:20;letter-spacing:.04em}
#help{position:fixed;bottom:14px;left:18px;font-size:11px;color:rgba(238,241,231,.34);
  z-index:20;font-family:var(--mono)}
#notes{position:fixed;left:0;right:0;bottom:0;background:rgba(8,12,10,.94);color:#cfe3d5;
  font-size:13.5px;line-height:1.55;padding:14px 22px;z-index:25;display:none;
  border-top:1px solid rgba(207,227,213,.18)}
#notes.on{display:block}
#grid{position:fixed;inset:0;background:#0d120f;z-index:30;display:none;overflow:auto;
  padding:28px;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px;
  align-content:start}
#grid.on{display:grid}
#grid .g{background:var(--bg);border-radius:6px;padding:12px;cursor:pointer;height:104px;
  overflow:hidden;border:2px solid transparent;position:relative}
#grid .g:hover{border-color:var(--accent)}
#grid .g b{font-family:var(--serif);font-size:13px;color:var(--brand);display:block;
  line-height:1.2}
#grid .g i{font-family:var(--mono);font-size:10px;color:var(--faint);font-style:normal;
  position:absolute;bottom:8px;left:12px}
@media print{
  html,body{overflow:visible;background:#fff}
  #stage{position:static;display:block}
  #deck{width:100%;height:auto;box-shadow:none;transform:none!important}
  .slide{display:flex!important;position:relative;page-break-after:always;
    width:1280px;height:720px;animation:none}
  #bar,#hud,#help,#notes,#grid{display:none!important}
}
</style>
</head>
<body>
<div id="bar"></div>
<div id="stage"><div id="deck">
__SLIDES__
</div></div>
<div id="hud"></div>
<div id="help">← → navigate · O overview · N notes · F fullscreen</div>
<div id="notes"></div>
<div id="grid"></div>
<script>
const slides=[...document.querySelectorAll('.slide')],N=slides.length;
const TITLES=__TITLES__;
let i=0,notesOn=false;
const bar=document.getElementById('bar'),hud=document.getElementById('hud'),
      notes=document.getElementById('notes'),grid=document.getElementById('grid');
function fit(){document.getElementById('deck').style.transform=
  'scale('+Math.min(innerWidth/1280,innerHeight/720)*0.96+')';}
function show(n){i=Math.max(0,Math.min(N-1,n));
  slides.forEach((s,k)=>s.classList.toggle('on',k===i));
  bar.style.width=((i+1)/N*100)+'%';
  hud.textContent=String(i+1).padStart(2,'0')+' / '+N;
  notes.textContent=slides[i].dataset.note||'—';
  history.replaceState(null,'','#'+(i+1));}
grid.innerHTML=slides.map((s,k)=>'<div class="g" data-k="'+k+'"><b>'+TITLES[k]+
  '</b><i>'+String(k+1).padStart(2,'0')+'</i></div>').join('');
grid.onclick=e=>{const g=e.target.closest('.g');
  if(g){show(+g.dataset.k);grid.classList.remove('on');}};
addEventListener('keydown',e=>{const k=e.key;
  if(k==='ArrowRight'||k==='PageDown'||k===' '){e.preventDefault();show(i+1);}
  else if(k==='ArrowLeft'||k==='PageUp'){e.preventDefault();show(i-1);}
  else if(k==='Home'){show(0);} else if(k==='End'){show(N-1);}
  else if(k==='o'||k==='O'){grid.classList.toggle('on');}
  else if(k==='n'||k==='N'){notesOn=!notesOn;notes.classList.toggle('on',notesOn);}
  else if(k==='f'||k==='F'){document.fullscreenElement?document.exitFullscreen():
    document.documentElement.requestFullscreen();}
  else if(k==='Escape'){grid.classList.remove('on');}
  else if(/^[1-9]$/.test(k)){show(parseInt(k)-1);}});
let x0=null;
addEventListener('touchstart',e=>x0=e.changedTouches[0].clientX,{passive:true});
addEventListener('touchend',e=>{if(x0===null)return;
  const dx=e.changedTouches[0].clientX-x0;
  if(Math.abs(dx)>50)show(i+(dx<0?1:-1));x0=null;},{passive:true});
addEventListener('resize',fit);
fit();show(parseInt(location.hash.slice(1))-1||0);
</script>
</body>
</html>
"""
