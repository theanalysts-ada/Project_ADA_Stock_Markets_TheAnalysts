/* The reader's bet at the opening, and its callback in the verdict. */
(function (NH) {
  const buttons = Array.from(document.querySelectorAll('#bet-buttons button'));
  const result = document.getElementById('bet-result');
  const text = document.getElementById('bet-text');
  const callback = document.getElementById('verdict-callback');

  buttons.forEach((button) => {
    button.addEventListener('click', () => {
      NH.bet = button.dataset.choice;
      NH.press(buttons, button);
      text.innerHTML = text.dataset[NH.bet];
      result.hidden = false;
      if (callback) callback.textContent = callback.dataset[NH.bet];
    });
  });

  const cost = document.getElementById('cost');
  const label = document.getElementById('cost-label');
  if (cost && label) {
    cost.addEventListener('input', () => { label.textContent = cost.value + ' bp'; });
  }
})(window.NightHeist);
