import {MDCTextField} from '@material/textfield';
import {MDCTabBar} from '@material/tab-bar';


const tabBar = new MDCTabBar(document.querySelector('.mdc-tab-bar'));

tabBar.listen('MDCTabBar:activated', (activatedEvent) => {
  console.log("event");
  document.querySelectorAll('.js-tab-content').forEach((element, index) => {
    if (index === activatedEvent.detail.index) {
      element.classList.remove('js-content-hidden');
    } else {
      element.classList.add('js-content-hidden');
    }
  });
});

const firstSentence = new MDCTextField(document.querySelector('.first-sentence'));
const secondSentence = new MDCTextField(document.querySelector('.second-sentence'));

//new MDCRipple(document.querySelector('.compare-sentences'));
