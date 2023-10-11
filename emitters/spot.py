import mitsuba as mi
import drjit as dr
import mitsuba
mi.set_variant('llvm_ad_rgb')

# mi.set_log_level(mi.LogLevel.Trace)

from ipdb import set_trace

class Spot(mi.Emitter):
    def __init__(self, props):
        super().__init__(props)

        # set flags
        self.m_flags = +mi.EmitterFlags.DeltaPosition
        # ---
        self.m_intensity = props.get("intensity", 1.0)

        # setup some other params
        cutoff_angle = props.get("cutoff_angle", 20.0)
        m_beam_width = props.get("beam_width", cutoff_angle * 3.0 / 4.0)
        self.m_cutoff_angle = (180.0/3.14)*(cutoff_angle)
        self.m_beam_width = (180.0/3.14) * (m_beam_width)
        self.m_inv_transition_width = 1.0 / (self.m_cutoff_angle - self.m_beam_width)
        self.m_cos_cutoff_angle = dr.cos(self.m_cutoff_angle)
        self.m_cos_beam_width = dr.cos(self.m_beam_width)
        self.m_uv_factor = dr.tan(self.m_cutoff_angle)

    def is_environment(self: mi.Emitter):
        return False
    
    def sampling_weight(self):
        pass

    def flags(self, active):
        return self.flags

    def falloff_curve(self, d, active):
        local_dir = dr.normalize(d) 
        cos_theta = local_dir.z
        beam_res = dr.select(
            cos_theta >= self.m_cos_cutoff_angle, 1.0,
            (self.m_cutoff_angle - dr.acos(cos_theta)) * self.m_inv_transition_width
        )

        return dr.select(
            cos_theta > self.m_cos_beam_width, beam_res, 0.0
        )

    def direction_to_uv(self, local_dir):
        return mi.Point2f(
            0.5 + 0.5 * local_dir.x/ (local_dir.z * self.m_uv_factor), 
            0.5 + 0.5 * local_dir.y/ (local_dir.z * self.m_uv_factor)
        )
    
    def sample_ray(self, time, wv_s, sp_s, dir_s, active):
        # sample direction
        local_dir = mi.warp.square_to_uniform_cone(sp_s, self.m_cos_cutoff_angle)
        pdf_dir = mi.warp.square_to_uniform_cone_pdf(local_dir, self.m_cos_cutoff_angle)

        # sample spec
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.time = time
        si.p = self.world_transform().translation()
        si.uv = self.direction_to_uv(local_dir)
        wavelengths, spec_weight = self.sample_wavelengths(si, wv_s, active)

        falloff = self.falloff_curve(local_dir, active)

        return mi.Ray3f(si.p, self.world_transform() * local_dir, time, wavelengths), \
            (spec_weight * falloff / pdf_dir)


    def sample_direction(self, it, sample, active):
        ds = dr.zeros(mi.DirectionSample3f)
        ds.p = self.world_transform().translation()
        ds.n = 0.0
        ds.uv = 0.0
        ds.pdf = 1.0
        ds.time = it.time
        ds.delta = True
        # ds.emitter = None #self
        ds.d = ds.p - it.p
        ds.dist = dr.norm(ds.d)
        inv_dist = dr.rcp(ds.dist)
        ds.d *= inv_dist
        # set_trace()
        local_d = self.world_transform().inverse() @ -ds.d

        falloff = self.falloff_curve(local_d, active)
        active &= falloff > 0.0

        si = dr.zeros(mi.SurfaceInteraction3f)
        si.t = 0.0
        si.time = it.time
        si.wavelengths = it.wavelengths
        si.p = ds.p 
        radiance = self.m_intensity.eval(si, active)

        return ds, (radiance & active ) * (falloff * dr.sqr(inv_dist))

    def pdf_direction(self, it, dir_s, active):
        return 0.0

    def eval_direction(self, it, ds, active):
        pass

    def sample_position(self, time, sample, active):
        center_dir = self.world_transform() * mi.ScalarVector3f(0.0, 0.0, 1.0)
        ps = mi.PositionSample3f(
            self.world_transform().translation(), center_dir,
            mi.Point2f(0.5), time, 1.0, True
        )   

        return ps, mi.Float(1.0)

    def sample_wavelength(self, si, sample, active):
        # TODO: sample_shifted will not work
        wav, weight = self.m_intensity.sample_spectrum(si, 
                mi.math.sample_shifted(sample), active)
        
        return wav, weight

    def eval(self, si, mask):
        return 0.0
    
    def bbox(self):
        p = self.world_transform().scalar() * mi.ScalarPoint3f(0.0)
        return mi.BoundingBox3f(p, p)
    



