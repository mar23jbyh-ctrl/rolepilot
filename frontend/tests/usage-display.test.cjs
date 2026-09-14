const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const ts=require('typescript');
const React=require('react');
const {renderToStaticMarkup}=require('react-dom/server');
function loadExports(name,overrides={}){
  const base=path.isAbsolute(name)?name:path.join(__dirname,'../src/components',name);
  const filename=fs.existsSync(base+'.tsx')?base+'.tsx':base+'.ts';
  const source=ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,target:ts.ScriptTarget.ES2020}}).outputText;
  const module={exports:{}};
  vm.runInNewContext(source,{module,exports:module.exports,localStorage:{getItem:()=>null},require:(name)=>Object.hasOwn(overrides,name)?overrides[name]:name==='lucide-react'?new Proxy({},{get:()=>()=>null}):name.startsWith('.')?loadExports(path.resolve(path.dirname(filename),name)):require(name)});
  return module.exports;
}
function load(name){return loadExports(name)[name];}
test('TopBar never renders usage even if old callers pass token props',()=>{
  const TopBar=load('TopBar');
  const unknown=renderToStaticMarkup(React.createElement(TopBar,{tokens:null,name:'synthetic',status:'interviewing'}));
  assert.doesNotMatch(unknown,/Token|部分用量未知/);
  assert.doesNotMatch(renderToStaticMarkup(React.createElement(TopBar,{tokens:0,name:'synthetic',status:'interviewing'})),/Token|部分用量未知/);
  assert.match(unknown,/synthetic|进行中/);
});

test('Sidebar shows history status without old token numbers',()=>{
  const Sidebar=load('Sidebar');
  const state={resumeText:'',jdText:'',sessionName:'',activeSessionId:'',resumeName:'',resumeMethod:'',loading:false,error:'',
    sessions:[{id:'synthetic',name:'Synthetic history',status:'completed',total_tokens:987654321}]};
  const html=renderToStaticMarkup(React.createElement(Sidebar,{state,onPatch:()=>{},onResume:()=>{},onJdImage:()=>{},onStart:()=>{},onView:()=>{},onDelete:()=>{},onRename:async()=>true}));
  assert.doesNotMatch(html,/987654321|tokens|部分用量未知/);
  assert.match(html,/Synthetic history|已完成/);
});
test('ReportView hides partial token totals and all monetary labels',()=>{
  const ReportView=load('ReportView');
  const html=renderToStaticMarkup(React.createElement(ReportView,{report:{overall_score:6,grade:'B',token_totals:{total:null,known_total:123,usage_complete:false},cost:null}}));
  assert.doesNotMatch(html,/Token|部分用量未知|已知 123|供应商账单|配置价格估算|0\.0000/);
});

test('ReportView retains score without usage, money or privacy notices',()=>{
  const ReportView=load('ReportView');
  const html=renderToStaticMarkup(React.createElement(ReportView,{report:{overall_score:6,grade:'B',token_totals:{total:123,known_total:123,usage_complete:true},cost:0.01}}));
  assert.doesNotMatch(html,/隐私提示|简历及回答发送|class="privacy"/);
  assert.doesNotMatch(html,/123|¥|配置价格估算|总 Token/);
  assert.match(html,/B/);
});

test('RolePilot branding and actual failed status appear in the header',()=>{
  const html=renderToStaticMarkup(React.createElement(load('TopBar'),{name:'Synthetic',status:'failed'}));
  assert.match(html,/RolePilot/);
  assert.match(html,/执行失败/);
  assert.doesNotMatch(html,/进行中|Token|¥/);
});

test('an OCR-only JD can enable start once its editable text is filled',()=>{
  const state={resumeText:'Synthetic resume',jdText:'Synthetic OCR JD',jdOcr:'Synthetic OCR JD',sessionName:'',activeSessionId:'',sessions:[],loading:false};
  const html=renderToStaticMarkup(React.createElement(load('Sidebar'),{state,onPatch:()=>{},onResume:()=>{},onJdImage:()=>{},onStart:()=>{},onView:()=>{},onDelete:()=>{},onRename:async()=>true}));
  assert.match(html,/RolePilot/);
  assert.doesNotMatch(html,/<button[^>]*disabled[^>]*>开始面试/);
});

test('legacy repeated answer evidence is deduplicated and is not an accuracy percentage',()=>{
  const report={overall_score:6,grade:'B',per_question:[{question_id:1,question:'Synthetic question',score:6,
    covered_points:['SQL','SQL'],missed_points:['抽样'],dimensions:{}}]};
  const html=renderToStaticMarkup(React.createElement(load('ReportView'),{report}));
  assert.doesNotMatch(html,/要点覆盖率|覆盖要点 2/);
  assert.match(html,/回答证据 1 条、待补充要点 1 条/);
  assert.equal((html.match(/<li>SQL<\/li>/g)||[]).length,1);
});

test('App renders an accessible in-page role editor rather than a native prompt',()=>{
  const state={activeSessionId:'synthetic',sessionName:'Synthetic',status:'interviewing',jobTitle:'Synthetic original role',
    resumeText:'',jdText:'',messages:[],sessions:[],loading:false,viewing:null,report:null,pendingAnswer:null};
  const callbacks=new Proxy({state},{get:(object,name)=>name==='state'?state:()=>{}});
  const App=loadExports(path.resolve(__dirname,'../src/App'),{'./hooks/useInterview':{useInterview:()=>callbacks},
    react:{...React,useState:(initial)=>[initial===null?{sessionId:'synthetic',text:'Synthetic revised role'}:initial,()=>{}]}}).default;
  const html=renderToStaticMarkup(React.createElement(App));
  assert.match(html,/aria-label="修正岗位名称"/);
  assert.match(html,/保存岗位名称|取消岗位修改/);
  assert.doesNotMatch(fs.readFileSync(path.resolve(__dirname,'../src/App.tsx'),'utf8'),/window\.prompt/);
});
