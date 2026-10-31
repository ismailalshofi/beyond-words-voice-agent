// Captures microphone audio, downsamples it to 16 kHz and posts 100 ms frames of PCM16.
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions?.targetRate || 16000;
    this.ratio = sampleRate / this.targetRate;
    this.frameSize = this.targetRate / 10;
    this.frame = new Int16Array(this.frameSize);
    this.index = 0;
    this.sum = 0;
    this.count = 0;
    this.position = 0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;

    for (let i = 0; i < input.length; i++) {
      // Average the input samples that fall into each output sample (cheap low-pass + decimate)
      this.sum += input[i];
      this.count++;
      this.position += 1;
      if (this.position >= this.ratio) {
        this.position -= this.ratio;
        const value = Math.max(-1, Math.min(1, this.sum / this.count));
        this.sum = 0;
        this.count = 0;
        this.frame[this.index++] = value < 0 ? value * 0x8000 : value * 0x7fff;
        if (this.index === this.frameSize) {
          this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
          this.frame = new Int16Array(this.frameSize);
          this.index = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
