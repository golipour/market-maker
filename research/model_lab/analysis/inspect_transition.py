import socket
import json
import datetime

def send_cmd(s, *args):
    buf = f"*{len(args)}\r\n"
    for a in args:
        a_str = str(a)
        buf += f"${len(a_str.encode())}\r\n{a_str}\r\n"
    s.sendall(buf.encode())

def parse_resp(f):
    prefix = f.read(1)
    if not prefix:
        return None
    if prefix == b'+':
        return f.readline().rstrip(b'\r\n').decode()
    elif prefix == b':':
        return int(f.readline().rstrip(b'\r\n'))
    elif prefix == b'$':
        length = int(f.readline().rstrip(b'\r\n'))
        if length == -1:
            return None
        data = f.read(length)
        f.read(2)
        return data.decode('utf-8', errors='replace')
    elif prefix == b'*':
        count = int(f.readline().rstrip(b'\r\n'))
        if count == -1:
            return None
        return [parse_resp(f) for _ in range(count)]
    elif prefix == b'-':
        err = f.readline().rstrip(b'\r\n').decode()
        raise RuntimeError(err)

def main():
    s = socket.socket()
    s.connect(('127.0.0.1', 6379))
    f = s.makefile('rb')

    send_cmd(s, 'XREVRANGE', 'mm.strategy.decisions', '+', '-', 'COUNT', '300')
    res = parse_resp(f)
    print(f"Fetched entries: {len(res)}")

    entries = []
    for entry in res:
        entry_id = entry[0]
        fields = entry[1]
        d = dict(zip(fields[0::2], fields[1::2]))
        payload_str = d.get('data') or d.get('payload')
        if not payload_str:
            continue
        try:
            ev = json.loads(payload_str)
            p = ev.get('payload', ev)
            ts_ms = int(entry_id.split('-')[0])
            dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000)
            entries.append((entry_id, dt, p))
        except Exception:
            pass

    entries.sort(key=lambda x: x[0])
    
    # Let's find entries between 09:23 and 09:32
    print(f"Earliest entry: {entries[0][1]} | Latest entry: {entries[-1][1]}")
    target_entries = [e for e in entries if datetime.time(9, 20) <= e[1].time() <= datetime.time(9, 32)]
    print(f"Target window entries: {len(target_entries)}")
    for entry_id, dt, p in target_entries:
        actor = p.get('actor_id')
        lat = p.get('latency_ms')
        opt = p.get('optimizer', {})
        timing = opt.get('timing', {})
        cfg = opt.get('config', {})
        epochs = timing.get('epochs_run')
        opt_ms = timing.get('optimize_ms')
        fwd_ms = timing.get('forward_ms')
        bwd_ms = timing.get('backward_ms')
        n_samples = cfg.get('n_samples')
        horizon = cfg.get('horizon')
        adam_epochs = cfg.get('adam_epochs')
        print(f"{dt.strftime('%H:%M:%S')} | {actor:<18} | lat={lat:<6.1f} | opt={opt_ms:<6.1f} | fwd={fwd_ms:<6.1f} | bwd={bwd_ms:<6.1f} | ep={epochs} (cfg: {adam_epochs}) | n_s={n_samples} | H={horizon}")

if __name__ == '__main__':
    main()
