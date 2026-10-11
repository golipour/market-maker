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

    send_cmd(s, 'XREVRANGE', 'mm.strategy.decisions', '+', '-', 'COUNT', '1000')
    res = parse_resp(f)
    print(f"Fetched entries: {len(res)}")

    rows = []
    print("Entry 0 keys:", len(res[0][1]))
    if len(res) > 0:
        fields = res[0][1]
        print("Sample fields:", fields[0:4])
    for entry in res:
        entry_id = entry[0]
        fields = entry[1]
        d = dict(zip(fields[0::2], fields[1::2]))
        payload_str = d.get('payload') or d.get('data') or d.get('body')
        if not payload_str and 'actor_id' in d:
            p = d
        elif payload_str:
            try:
                p = json.loads(payload_str)
            except Exception:
                p = d
        else:
            continue
        try:
            # If enveloped, payload might be nested under 'payload'
            if 'payload' in p and isinstance(p['payload'], dict):
                inner = p['payload']
            else:
                inner = p
            actor = inner.get('actor_id') or p.get('actor_id')
            ts = inner.get('t_end_ms') or p.get('t_end_ms')
            lat = inner.get('latency_ms') or 0.0
            opt = inner.get('optimizer', {})
            timing = opt.get('timing', {})
            opt_ms = timing.get('optimize_ms') or 0.0
            fwd_ms = timing.get('forward_ms') or 0.0
            bwd_ms = timing.get('backward_ms') or 0.0
            obj_ms = timing.get('objective_ms') or 0.0
            epochs = timing.get('epochs_run') or 0
            best_ep = timing.get('best_epoch') or 0
            rows.append((entry_id, ts, actor, lat, opt_ms, fwd_ms, bwd_ms, obj_ms, epochs, best_ep, opt.get('config', {})))
        except Exception as e:
            pass

    rows.sort(key=lambda x: x[0])
    
    # Bucket by 5-minute intervals
    from collections import defaultdict
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ts_ms = int(r[0].split('-')[0])
        dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000)
        # bucket key: YYYY-MM-DD HH:(MM//5 * 5)
        bucket_key = dt.strftime('%H:') + f"{(dt.minute // 5) * 5:02d}"
        actor = r[2]
        buckets[bucket_key][actor].append(r)

    print("\n=== 5-MINUTE BUCKETS SUMMARY ===")
    print(f"{'Interval':<10} | {'Actor':<18} | {'Count':<5} | {'Avg Lat':<8} | {'Avg Opt':<8} | {'Avg Fwd':<8} | {'Avg Bwd':<8} | {'Avg Obj':<8}")
    print('-' * 90)
    for b_key in sorted(buckets.keys()):
        for actor in sorted(buckets[b_key].keys()):
            items = buckets[b_key][actor]
            n = len(items)
            avg_lat = sum(x[3] for x in items) / n
            avg_opt = sum(x[4] for x in items) / n
            avg_fwd = sum(x[5] for x in items) / n
            avg_bwd = sum(x[6] for x in items) / n
            avg_obj = sum(x[7] for x in items) / n
            print(f"{b_key:<10} | {actor:<18} | {n:<5} | {avg_lat:<8.1f} | {avg_opt:<8.1f} | {avg_fwd:<8.1f} | {avg_bwd:<8.1f} | {avg_obj:<8.1f}")


if __name__ == '__main__':
    main()
