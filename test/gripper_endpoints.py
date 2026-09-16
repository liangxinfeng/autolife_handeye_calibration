"""Derive visualization endpoints from finger mesh clearance (not motor calibration)."""
exec(open(__file__.replace('gripper_endpoints.py','gripper_geometry.py')).read().split('axis=None')[0])
for side in ('Left','Right'):
    def clouds(angle):
        q={f'Joint_{side}_Gripper':angle}
        for j in root.findall('joint'):
            m=j.find('mimic')
            if m is not None:q[j.get('name')]=q.get(m.get('joint'),0)*float(m.get('multiplier','1'))+np.degrees(float(m.get('offset','0')))
        tf=model.link_transforms(q);out=[]
        for hand in ('Left','Right'):
            link=f'Link_{side}_Gripper_{hand}_Finger';v=next(v for v in model.visuals if v.link==link)
            vertices,_=_load_binary_stl(v.mesh_path);t=tf[link]@v.origin
            out.append(vertices@t[:3,:3].T+t[:3,3])
        return out
    c=clouds(0);axis=c[1].mean(axis=0)-c[0].mean(axis=0);axis/=np.linalg.norm(axis)
    def gap(angle):
        c=clouds(angle);return np.min(c[1]@axis)-np.max(c[0]@axis)
    def solve(target):
        lo,hi=-30.,40.
        for _ in range(25):
            mid=(lo+hi)/2
            if gap(mid)<target:lo=mid
            else:hi=mid
        return (lo+hi)/2
    print(side,'closed',solve(0),'open',solve(.095))
