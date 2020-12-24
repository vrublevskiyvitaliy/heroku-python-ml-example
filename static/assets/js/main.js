'use strict';


mdc.ripple.MDCRipple.attachTo(document.querySelector('.foo-button'));
const tab = mdc.ripple.MDCTabBar.attachTo(document.querySelector('.mdc-tab-bar'));

tab.listen('MDCTabBar:activated', (activatedEvent) => {
  document.querySelectorAll('.tab-content').forEach((element, index) => {
    console.log(index);
    if (index === activatedEvent.detail.index) {
      element.classList.remove('hidden');
    } else {
      element.classList.add('hidden');
    }
  });
});