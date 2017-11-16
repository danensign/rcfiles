console.log('Loading custom.js');
require(['notebook/js/codecell'], function (codecell) {
    codecell.CodeCell.options_default.cm_config.autoCloseBrackets = false;
});
jQuery('#header-container').hide();
