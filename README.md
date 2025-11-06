# INSALO
Hybrid closed loop insulin pump system with influence by external factors.

# Abstract 
Type 1 Diabetes, or diabetes mellitus, affects over 8.4 million people worldwide, with
an expected growth to 13.5-17.4 million. Although many of these diagnoses affect
men, it’s undeniable that the impact on females is far more detrimental than that to
their counterparts. In fact, women have a 37% more chance of suffering fatal
consequences due to type 1 diabetes in comparison to men.
The existence of a menstrual cycle in terms of diabetic management has proven to
be a contributing factor to the unprecedented changes within diabetic management.
Currently, no technology exists to combat this, as the most recent developments of
hybrid closed loop insulin pumps work on a 2–6-week adaptive period of
background insulin needs. However, with the effects of estrogen, progesterone and
other hormones which naturally occur in female bodies, insulin management
becomes increasingly difficult, as reported by the National Library of Medicine.
Moreover, these shifts throughout a month can impact body fat percentages,
appetite and other factors which insulin pumps at the moments do not account for.
Another blaring limitation of insulin pumps in the modern age is the necessity to
constantly change infusion set sites, impairing users’ ability to freely move about
without being in concern of infusion set sites staying intact. The other main reason
for the consistent movement in set sites is the trepidation of lipohypertrophy, which
can lead to increased insulin insensitivity, serious complications and overall take
months/years to recover from. In younger patients, frequently changing a set site
can also be painful and at times traumatizing, and for athletes who participate in
high contact sports, infusion set sites can easily be damaged or fall out during
intense moments. This proves that, although the current system of regular infusion
set site changes has been efficient, reworking some factors of it would prove to have
more positive results and become substantially more accessible for all
demographics.

# Design
INSALO features both hardware and software components which add to its adaptability and allows
for effective diabetes management for all users.

  # Hardware components (HALO Dock)
  Subdermal microfluid “diffusion” chamber
Made of medical grade silicon (ISO 10993), containing small chamber inside before insulin moves
out of ~10–50 µm holes/microneedles into nearby capillaries. Implanted chamber is porous with
dimensions 16mm diameter x 2mm thickness. Allows for approx. 200-400 holes per mm2 ,
providing sufficient surface area : volume ratio and effective insulin transport. Chamber to sit under
dermis layer of the skin (approx. 1-2mm depth)

  Internal cannula and semi-permanent port
Titanium or medical grade silicon cannula (2mm), decided by patient per personal preferences
and/or lifestyle compatibility. Cannula to be fused into the microfluid chamber and connected to a
medical grade silicon or titanium port. Internal port (9.2mm in length, elliptical shape) to be
accessible from external environment, however, must stay closed unless changing the external
connection site.

  External connection site (Dock)
Polyetheretherketone (PEEK) plastic cover, lined with rubber O rings around the connection port
for waterproofing and a premium option over other plastics. Connection site to be changed by
medical professionals, as it involves the internal hardware. The Dock will contain two holding
elements, one for the insulin reservoir and one for the phone-to-site Bluetooth transmitter.
Dimensions to be 50mm x 18mm x 6mm, with a general curvature to the shape to allow for smooth
integration with everyday life.

  Reservoir
Reservoir with maximum capacity of 500U (5mL), which can be changed by the user at any time.

  Transmitter unit
Transmitter to have rechargeable battery, and have Bluetooth compatibility for ultimate control of
the device. INSALO kits to come with two transmitters, so that there is always a backup for when
the other is charging. Transmitter will control only how much insulin is pushed through the pump
unit, gently pushing on the reservoir into the cannula to control insulin secretion.

  Sticker overlay
A sticker (IV3000) overlay, to cover the transmitter and reservoir on top of the Dock. This will
ensure stability and can bring in an element of personalisation for smoother integration into
everyday life.


  # Software features
The INSALO system will operate on a hybrid closed machine learning-assisted loop system,
allowing the user to maintain control over individual boluses and settings whilst maintaining a
smart option for optimal diabetes management, whilst considering general health and external
factors. It will be available for both handheld (separate) devices (applicable for younger users or
those who do not have access to a mobile phone) and for mobile phone usage.

  User App (mobile or handheld)
➢ Mobile app and handheld pump available for running software
➢ Show real time glucose graph derived from chosen CGM
➢ Show summary of recent events and health recaps
➢ Send alerts for highs, lows and reminders for future
➢ Allow user to share data and information with their chosen people
➢ Conduct nightly uploads into the cloud for easy access for doctors/endocrinologists to
consult during appointments

  Hybrid Closed Loop
➢ Read CGM data from supported devices (eg. Dexcom, Freestyle Libre)
➢ Predict future glucose trends with assistance from machine learning based on external
factors as well as past BGL tendencies
➢ Adjust basal insulin automatically through “microboluses” of up to 0.025U-0.08U depending
on circumstances
➢ Deliver automatic correction boluses of anything >0.8U
➢ Monitor for drops in blood sugar and stop delivery for predicted hypoglycaemia
➢ Log all dosing actions done by the user or system
User Awareness
➢ Collect and store user data such as height, weight, current illnesses and other physical
factors
➢ Collect and store user data regarding mental health status and sleep patterns, as well as
appetite or amount of food eaten
➢ Track menstrual cycle phases (for female bodied individuals)
➢ Adjust insulin sensitivity, carb ratios and BGL targets based on current data such as
mood/stress levels, sicknesses, weight trends, menstrual phases and data derived from
wearables

  Wearable Integration
➢ Connect to Apple Health/Google Fit and other health tracking services
➢ Collect workout sessions, activity level and calories burned data
➢ Collect sleep quality data
➢ By choice of user, promote fitness/healthy habits

