from __future__ import annotations
import argparse
import sys
from . import metrics
from .image_io import load_image, save_image
from .methods import get_method, list_methods
from .pipeline import Pipeline

def _make_pipeline(args) -> Pipeline:
    method_kwargs = {}
    if args.key_seed is not None:
        method_kwargs['key'] = args.key_seed
    if args.method in ('dct', 'rdct') and args.step is not None:
        method_kwargs['step'] = args.step
    if args.method == 'lsb' and args.bpc is not None:
        method_kwargs['bits_per_channel'] = args.bpc
    if args.method == 'amx' and args.k is not None:
        method_kwargs['k'] = args.k
    carrier = get_method(args.method, **method_kwargs)
    return Pipeline(carrier=carrier, passphrase=args.key, frame=not args.no_frame)

def cmd_hide(args) -> int:
    image = load_image(args.input)
    pipe = _make_pipeline(args)
    data = args.message.encode('utf-8') if args.message else sys.stdin.buffer.read()
    cap = pipe.capacity_bytes(image)
    if len(data) > cap:
        print(f'error: payload {len(data)} B exceeds capacity {cap} B', file=sys.stderr)
        return 2
    stego = pipe.hide(image, data)
    save_image(stego, args.output)
    print(f'hid {len(data)} B in {args.output} (PSNR {metrics.psnr(image, stego):.2f} dB, utilisation {len(data) / cap:.1%})')
    return 0

def cmd_reveal(args) -> int:
    image = load_image(args.input)
    pipe = _make_pipeline(args)
    data = pipe.reveal(image)
    if args.output:
        with open(args.output, 'wb') as fh:
            fh.write(data)
        print(f'wrote {len(data)} B to {args.output}')
    else:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.write(b'\n')
    return 0

def cmd_analyze(args) -> int:
    cover = load_image(args.input)
    stego = load_image(args.stego)
    print(f'MSE        : {metrics.mse(cover, stego):.4f}')
    print(f'PSNR       : {metrics.psnr(cover, stego):.2f} dB')
    print(f'SSIM       : {metrics.ssim(cover, stego):.6f}')
    print(f'chi2 cover : {metrics.chi_square_lsb(cover):.4f}')
    print(f'chi2 stego : {metrics.chi_square_lsb(stego):.4f}')
    return 0

def cmd_capacity(args) -> int:
    image = load_image(args.input)
    for name in [args.method] if args.method else list_methods():
        carrier = get_method(name)
        bits = carrier.capacity_bits(image)
        print(f'{name:5s}: {bits:>10d} bits  ({bits // 8:>9d} B)')
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='stegopipe', description='Image steganography pipeline v2')
    sub = p.add_subparsers(dest='command', required=True)

    def add_common(sp):
        sp.add_argument('-i', '--input', required=True, help='input image path')
        sp.add_argument('--method', default='lsb', choices=list_methods())
        sp.add_argument('-k', '--key', default=None, help='passphrase (enables encryption)')
        sp.add_argument('--key-seed', type=int, default=None, help='PRNG seed for embedding order')
        sp.add_argument('--step', type=float, default=None, help='DCT/rdct quantisation step')
        sp.add_argument('--bpc', type=int, default=None, help='LSB bits per channel (1-4)')
        sp.add_argument('--k', type=int, default=None, help='amx matrix-embedding parameter k')
        sp.add_argument('--no-frame', action='store_true', help='disable integrity frame')
    h = sub.add_parser('hide', help='embed a message')
    add_common(h)
    h.add_argument('-o', '--output', required=True, help='output stego image (lossless)')
    h.add_argument('-m', '--message', help='message text (else read stdin bytes)')
    h.set_defaults(func=cmd_hide)
    r = sub.add_parser('reveal', help='extract a message')
    add_common(r)
    r.add_argument('-o', '--output', help='write bytes to file (else stdout)')
    r.set_defaults(func=cmd_reveal)
    a = sub.add_parser('analyze', help='compare cover vs stego')
    a.add_argument('-i', '--input', required=True, help='cover image')
    a.add_argument('-s', '--stego', required=True, help='stego image')
    a.set_defaults(func=cmd_analyze)
    c = sub.add_parser('capacity', help='report carrier capacity')
    c.add_argument('-i', '--input', required=True)
    c.add_argument('--method', default=None, choices=list_methods())
    c.set_defaults(func=cmd_capacity)
    return p

def main(argv: list[str] | None=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
if __name__ == '__main__':
    raise SystemExit(main())
