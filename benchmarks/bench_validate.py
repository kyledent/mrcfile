"""validate() on an already-open file, isolating it from read cost."""

import io

import _harness as h

import mrcfile


def main():
    path = h.fixture("vol_f32.mrc")
    h.warm(path)
    mrc = mrcfile.open(path)
    sink = io.StringIO()

    def validate():
        sink.truncate(0)
        return mrc.validate(print_file=sink)

    h.emit("validate", {"validate_128MiB": h.measure(validate)})
    mrc.close()


main()
