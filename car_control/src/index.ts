import jQuery from 'jquery';

// @ts-ignore
window.jQuery = window.$ = jQuery;
import 'semantic-ui-css/semantic';

//===============================================================

// @ts-ignore
import Vue from 'vue';
import App from '/templates/App.vue';

let app = new Vue({
  el: '#app',
  render: h => h(App),
});

//===============================================================
// Application main context

import context from '/lib/context';
// @ts-ignore
window.context = context;

//===============================================================
// Application initialization

context.setEnv('cases/case01.txt');
context.emit('changeModuleTab', 'test');
context.setModule('test');
