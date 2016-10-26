
def parse_date(sdate):
    """Parses: 'y-m-d' --> (y,m,d)
        'y-m' --> (y,m)
        'y' --> (y,)"""
    sdate = sdate.strip()
    if len(sdate) == 0:
        return Date()
    else:
        return Date(*(int(x) for x in sdate.split('-')))

def _parse_delta(delta):
    if isinstance(delta, tuple):
        delta = (0,0) + delta
        return delta[0], delta[1], delta[2]
    else:
        return (0,0,delta,)    # Days

class Date(tuple):
    """Dates are represented by:
       (Y,M,D) tuples; or just (Y,M) or (Y,)"""

    def __new__(cls, *args):
        return super(Date,cls).__new__(cls, args)

    @property
    def year(self):
        return self[0]

    @property
    def month(self):
        return self[1]

    @property
    def day(self):
        return self[2]

    def __add__(self, delta):
        y,m,d = self
        dy,dm,dd = _parse_delta(delta)
        y += dy + dm // 12
        m += dm % 12
        return jday_to_date(date_to_jday((y,m,d)) + dd)

    def __iadd__(self, delta):
        return self + delta

    def __sub__(self, delta):
        negdelta = tuple(-x for x in delta)
        return self + negdelta

    def __isub__(self, delta):
        negdelta = tuple(-x for x in delta)
        return self + negdelta

def date_to_jday(date):
    """This is the Julian Day Number for the beginning of the date in
    question at 0 hours, Greenwich time. Note that this always gives
    you a half day extra. That is because the Julian Day begins at
    noon, Greenwich time. This is convenient for astronomers (who
    until recently only observed at night), but it is confusing.

    Taken from: http://quasar.as.utexas.edu/BillInfo/JulianDatesG.html

    NOTE: This actually returns jday-.5
    """

    date = tuple(date) + (1,1)    # Extra long now
    Y = date[0]
    M = date[1]
    D = date[2]

    if M < 3:
        Y -= 1
        M += 12

    A = int(Y // 100)
    B = A // 4
    C = 2-A+B
    # E = int(365.25 * (Y+4716))
    E = (36525 * (Y+4716)) // 100
    # F = int(30.6001 * (M+1))
    F = (306001 * (M+1)) // 10000
    # JD= C+D+E+F-1524.5
    JD= C+D+E+F-1525

    return JD

def jday_to_date(JD):
    """To convert a Julian Day Number to a Gregorian date, assume that it
    is for 0 hours, Greenwich time (so that it ends in 0.5). Do the
    following calculations, again dropping the fractional part of all
    multiplicatons and divisions. Note: This method will not give
    dates accurately on the Gregorian Proleptic Calendar, i.e., the
    calendar you get by extending the Gregorian calendar backwards to
    years earlier than 1582. using the Gregorian leap year rules. In
    particular, the method fails if Y<400. Thanks to a correspondent,
    Bo Du, for some ideas that have improved this calculation."""

    # Q = JD+0.5
    Q = JD+1
    # Z = Integer part of Q
    Z = Q
    #W = int((Z - 1867216.25)/36524.25)
    W = (Z*100 - 186721625)//3652425
    X = W//4
    A = Z+1+W-X
    B = A+1524
    #C = int((B-122.1)/365.25)
    C = (B*100-12210) // 36525
    # D = int(365.25*C)
    D = (36525*C) // 100
    E = ((B-D)*10000) // 306001
    F = (306001*E) // 10000

    day =  B-D-F+(Q-Z)

    # month = E-1 or E-13 (must get number less than or equal to 12)
    month = E-13 if E>13 else E-1

    # year = C-4715 (if Month is January or February) or C-4716 (otherwise)
    year = C - (4715 if month <= 2 else 4716)

    return Date(year,month,day)
